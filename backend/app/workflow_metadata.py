"""Durable workflow metadata contracts; never graph checkpoints or replay payloads."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import Field, StringConstraints
from .domain import Record


Fingerprint = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Provider = Literal["local_scripted", "bedrock"]


class WorkflowMetadataStatus(StrEnum):
    RUNNING = "RUNNING"
    RESUMING = "RESUMING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REVIEWED = "REVIEWED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


ACTIVE_WORKFLOW_STATUSES = frozenset({
    WorkflowMetadataStatus.RUNNING,
    WorkflowMetadataStatus.RESUMING,
    WorkflowMetadataStatus.REVIEW_REQUIRED,
})

WORKFLOW_STATUS_TRANSITIONS = {
    WorkflowMetadataStatus.RUNNING: frozenset({
        WorkflowMetadataStatus.REVIEW_REQUIRED, WorkflowMetadataStatus.COMPLETED,
        WorkflowMetadataStatus.FAILED, WorkflowMetadataStatus.RECOVERY_REQUIRED,
    }),
    WorkflowMetadataStatus.REVIEW_REQUIRED: frozenset({
        WorkflowMetadataStatus.RESUMING, WorkflowMetadataStatus.REVIEWED,
        WorkflowMetadataStatus.RECOVERY_REQUIRED,
    }),
    WorkflowMetadataStatus.RESUMING: frozenset({
        WorkflowMetadataStatus.REVIEW_REQUIRED, WorkflowMetadataStatus.REVIEWED,
        WorkflowMetadataStatus.FAILED, WorkflowMetadataStatus.RECOVERY_REQUIRED,
    }),
    WorkflowMetadataStatus.REVIEWED: frozenset(),
    WorkflowMetadataStatus.COMPLETED: frozenset(),
    WorkflowMetadataStatus.FAILED: frozenset(),
    WorkflowMetadataStatus.RECOVERY_REQUIRED: frozenset(),
}


class WorkflowRunReservation(Record):
    workflow_id: UUID
    run_id: UUID
    case_id: UUID
    owner_user_id: UUID
    request_id: UUID
    request_fingerprint: Fingerprint
    provider: Provider


class WorkflowRunMetadata(WorkflowRunReservation):
    status: WorkflowMetadataStatus
    created_at: datetime
    updated_at: datetime


class WorkflowReservationResult(Record):
    run: WorkflowRunMetadata
    created: bool
    replayed: Literal[False] = False


class WorkflowRunEvent(Record):
    workflow_id: UUID
    sequence: int = Field(ge=1)
    kind: Literal["CREATED", "STATUS_CHANGED", "RECOVERY_REQUIRED"]
    previous_status: WorkflowMetadataStatus | None = None
    status: WorkflowMetadataStatus
    created_at: datetime


class WorkflowMetadataRepository(Protocol):
    def reserve(self, reservation: WorkflowRunReservation) -> WorkflowReservationResult: ...
    def get(self, workflow_id: UUID) -> WorkflowRunMetadata: ...
    def get_by_run_id(self, run_id: UUID) -> WorkflowRunMetadata: ...
    def get_by_request(self, owner_user_id: UUID, request_id: UUID) -> WorkflowRunMetadata: ...
    def active_runs(self) -> tuple[WorkflowRunMetadata, ...]: ...
    def visible_runs_for_case(self, case_id: UUID, owner_user_id: UUID | None,
                              offset: int, limit: int) -> tuple[WorkflowRunMetadata, ...]: ...
    def transition(self, workflow_id: UUID, *, owner_user_id: UUID, case_id: UUID,
                   expected_status: WorkflowMetadataStatus,
                   status: WorkflowMetadataStatus) -> WorkflowRunMetadata: ...
    def require_recovery(self, workflow_id: UUID, *, owner_user_id: UUID,
                         case_id: UUID,
                         expected_status: WorkflowMetadataStatus) -> WorkflowRunMetadata: ...
    def events(self, workflow_id: UUID) -> tuple[WorkflowRunEvent, ...]: ...
