"""Single-process, bounded HTTP adapter over the existing HITL and memory services."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from threading import RLock
from uuid import UUID, uuid4

from .agent import AgentRequest
from .frontend_contracts import (
    AuditPage, AuditSummary, EventPage, PendingReviewPage, PendingReviewRun, RunEvent, RunPage, RunSnapshot,
)
from .hitl import HITLWorkflow, HumanDecision, ResumeRejected
from .knowledge import RetrievalResult
from .llm import StructuredLLM
from .observability import ERRORS, TOOLS
from .proposals import ProposalStatus, ResolutionProposal
from .security import Authorization, Role, SecurityError, current_identity, require_roles


def now():
    return datetime.now(timezone.utc)


def safe_error(code):
    return code if code in ERRORS else "WORKFLOW_ERROR" if code else None


class FrontendError(Exception):
    def __init__(self, status, code):
        self.status = status
        self.code = code
        super().__init__(code)


@dataclass
class _Run:
    snapshot: RunSnapshot
    owner: UUID
    fingerprint: str
    workflow: HITLWorkflow | None = None
    events: deque = field(default_factory=deque)


class FrontendRuntime:
    # No run eviction: deduplication keys and checkpoints live for this process.
    # Once full, refuse new starts rather than silently replaying an evicted write.
    def __init__(self, cases, tools, proposals, *, local_provider_factory=None, memory=None,
                 max_runs=128, max_active=2, event_capacity=128):
        if not (1 <= max_runs <= 1024 and 1 <= max_active <= 8 and 1 <= event_capacity <= 1024):
            raise ValueError("Invalid local runtime bounds")
        self.instance_id = uuid4()
        self.provider_factory = local_provider_factory
        self.memory = memory
        self._cases, self._tools, self._proposals = cases, tools, proposals
        self._max_runs, self._max_active, self._event_capacity = max_runs, max_active, event_capacity
        self._runs, self._requests = {}, {}
        self._active = 0
        self._closed = False
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_active, thread_name_prefix="novamind-run")

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)

    def check_instance(self, instance_id):
        if instance_id != self.instance_id:
            raise FrontendError(409, "INSTANCE_CHANGED_DO_NOT_REPLAY")

    def _authorized(self, run_id):
        identity = current_identity()
        run = self._runs.get(run_id)
        if run is None:
            if identity.role == Role.CUSTOMER:
                raise SecurityError()
            raise FrontendError(404, "RUN_NOT_FOUND_OR_RESTARTED")
        Authorization(self._cases).case(run.snapshot.case_id)
        if identity.role == Role.CUSTOMER and identity.user_id != run.owner:
            raise SecurityError()
        return run

    def _publish(self, run, kind, state, **fields):
        timestamp = now()
        event = RunEvent(run_id=run.snapshot.run_id, sequence=run.snapshot.last_sequence + 1,
                         timestamp=timestamp, kind=kind, state=state, **fields)
        run.events.append(event)
        run.snapshot = run.snapshot.model_copy(update={"last_sequence": event.sequence, "updated_at": timestamp})

    def _transition(self, run, node, phase, state):
        with self._lock:
            run.snapshot = run.snapshot.model_copy(update={"workflow_id": state.workflow_id,
                "conversation_id": state.agent.request.conversation_id, "review": state.review})
            self._publish(run, "NODE", phase, node=node)
            if node not in ("load_case", "execute_tool"):
                return
            entry = state.agent.staged_event
            if node == "load_case" and state.agent.history:
                entry = state.agent.history[0]
            call = state.agent.decision if node == "execute_tool" else None
            name = getattr(call, "name", None) if node == "execute_tool" else "get_case"
            if name not in TOOLS:
                return
            # Publish only actual dispatcher results. A node can reject a call
            # before invoking a tool (case/proposal budgets); that is not tool activity.
            if phase == "COMPLETED" and entry is not None and entry.result is not None:
                result = entry.result
                data = result.data if result is not None and result.ok else None
                code = entry.error or (result.error.code if result is not None and not result.ok else None)
                self._publish(run, "TOOL", phase, tool=name, ok=bool(result and result.ok),
                    error=safe_error(code), proposal_id=data.id if isinstance(data, ResolutionProposal) else None,
                    evidence=data.hits if isinstance(data, RetrievalResult) else (),
                    retrieval_method=data.method if isinstance(data, RetrievalResult) else None)

    def _reserve(self):
        if self._closed or self._active >= self._max_active:
            raise FrontendError(503, "RUN_CAPACITY_UNAVAILABLE")
        self._active += 1

    def start(self, body, *, trace_id=None):
        identity = current_identity()
        Authorization(self._cases).case(body.case_id)
        self.check_instance(body.instance_id)
        fingerprint = sha256(body.model_dump_json().encode()).hexdigest()
        key = (identity.user_id, body.request_id)
        with self._lock:
            if key in self._requests:
                run = self._authorized(self._requests[key])
                if fingerprint != run.fingerprint:
                    raise FrontendError(409, "REQUEST_ID_CONFLICT")
                return run.snapshot
            if self.provider_factory is None:
                raise FrontendError(503, "PROVIDER_UNAVAILABLE")
            if len(self._runs) >= self._max_runs:
                raise FrontendError(503, "RUN_RETENTION_FULL")
            if body.conversation_id is not None:
                if self.memory is None:
                    raise FrontendError(503, "CONVERSATION_UNAVAILABLE")
                self.memory.load(body.conversation_id, body.case_id)
            self._reserve()
            timestamp = now()
            run = _Run(RunSnapshot(instance_id=self.instance_id, run_id=uuid4(), case_id=body.case_id,
                conversation_id=body.conversation_id, status="RUNNING", created_at=timestamp, updated_at=timestamp,
                trace_id=trace_id),
                identity.user_id, fingerprint, events=deque(maxlen=self._event_capacity))
            self._runs[run.snapshot.run_id] = run
            self._requests[key] = run.snapshot.run_id
            self._publish(run, "STATE", "RUNNING")
            request = AgentRequest.model_validate(body.model_dump(exclude={"instance_id", "request_id"}))
            self._submit(run, request, resume=False)
            return run.snapshot

    def _submit(self, run, payload, *, resume):
        try:
            self._executor.submit(copy_context().run, self._execute, run, payload, resume)
        except Exception:
            self._active -= 1
            run.snapshot = run.snapshot.model_copy(update={"status": "FAILED", "error": "WORKFLOW_ERROR"})
            self._publish(run, "STATE", "FAILED", error="WORKFLOW_ERROR")

    def _execute(self, run, payload, resume):
        try:
            if not resume:
                run.workflow = HITLWorkflow(StructuredLLM(self.provider_factory()), self._tools,
                    self._proposals, memory=self.memory,
                    on_transition=lambda *args: self._transition(run, *args))
            result = run.workflow.resume(payload) if resume else run.workflow.start(payload)
            with self._lock:
                if isinstance(result, ResumeRejected):
                    run.snapshot = run.snapshot.model_copy(update={"status": "REVIEW_REQUIRED", "error": result.error})
                else:
                    run.snapshot = run.snapshot.model_copy(update={
                        "workflow_id": result.workflow_id, "status": result.status,
                        "conversation_id": result.conversation_id or run.snapshot.conversation_id,
                        "review": result.review or run.snapshot.review, "reviewed_status": result.reviewed_status,
                        "message": result.message, "error": safe_error(result.error)})
                self._publish(run, "STATE", run.snapshot.status, error=run.snapshot.error)
        except Exception:
            with self._lock:
                run.snapshot = run.snapshot.model_copy(update={"status": "FAILED", "error": "WORKFLOW_ERROR",
                    "message": "Workflow stopped. Inspect current records; do not replay. No action was executed."})
                self._publish(run, "STATE", "FAILED", error="WORKFLOW_ERROR")
        finally:
            with self._lock:
                self._active -= 1

    def snapshot(self, run_id):
        with self._lock:
            return self._authorized(run_id).snapshot

    def events(self, run_id, after, limit):
        with self._lock:
            run = self._authorized(run_id)
            if after > run.snapshot.last_sequence:
                raise FrontendError(409, "EVENT_CURSOR_AHEAD")
            first = run.events[0].sequence
            events = tuple(event for event in run.events if event.sequence > after)[:limit]
            cursor = events[-1].sequence if events else after
            return EventPage(snapshot=run.snapshot, events=events, next_after=cursor,
                has_more=cursor < run.snapshot.last_sequence, first_available_sequence=first,
                dropped_events=first - 1, gap=after < first - 1)

    def list_for_case(self, case_id, offset, limit):
        Authorization(self._cases).case(case_id)
        identity = current_identity()
        with self._lock:
            records = [run.snapshot for run in self._runs.values() if run.snapshot.case_id == case_id
                       and (identity.role != Role.CUSTOMER or run.owner == identity.user_id)]
            return RunPage(items=tuple(records[offset:offset + limit]),
                next_offset=offset + limit if offset + limit < len(records) else None)

    def pending_reviews(self):
        require_roles(Role.REVIEWER, Role.ADMIN)
        authorization = Authorization(self._cases, self._proposals)
        with self._lock:
            items = []
            for run in self._runs.values():
                snapshot, review = run.snapshot, run.snapshot.review
                if snapshot.status != "REVIEW_REQUIRED" or review is None:
                    continue
                support_case = authorization.case(snapshot.case_id)
                proposal = authorization.proposal(review.proposal_id)
                if (proposal.status != ProposalStatus.PENDING_REVIEW
                        or proposal.case_id != snapshot.case_id or review.case_id != snapshot.case_id
                        or review.workflow_id != snapshot.workflow_id):
                    continue
                items.append(PendingReviewRun(run_id=snapshot.run_id, case_id=snapshot.case_id,
                    case_subject=support_case.subject, workflow_id=snapshot.workflow_id,
                    updated_at=snapshot.updated_at, review=review))
            return PendingReviewPage(instance_id=self.instance_id, snapshot_at=now(), items=tuple(items))

    def decide(self, run_id, body, *, trace_id=None):
        require_roles(Role.REVIEWER, Role.ADMIN)
        self.check_instance(body.instance_id)
        with self._lock:
            run = self._authorized(run_id)
            if run.snapshot.status != "REVIEW_REQUIRED":
                raise FrontendError(409, "NOT_PAUSED")
            review = run.snapshot.review
            if review is None or any(getattr(body, key) != getattr(review, key)
                                     for key in ("workflow_id", "case_id", "proposal_id", "review_id")):
                raise FrontendError(409, "REVIEW_MISMATCH")
            proposal = Authorization(self._cases, self._proposals).proposal(body.proposal_id)
            if proposal.status != ProposalStatus.PENDING_REVIEW:
                raise FrontendError(409, "ALREADY_REVIEWED")
            self._reserve()
            run.snapshot = run.snapshot.model_copy(update={"status": "RESUMING", "error": None, "trace_id": trace_id})
            self._publish(run, "STATE", "RESUMING")
            self._submit(run, HumanDecision.model_validate(body.model_dump(exclude={"instance_id"})), resume=True)
            return run.snapshot

    def audit(self, run_id, after, limit):
        require_roles(Role.REVIEWER, Role.ADMIN)
        with self._lock:
            run = self._authorized(run_id)
            if run.snapshot.status in ("RUNNING", "RESUMING"):
                raise FrontendError(409, "AUDIT_PENDING_USE_EVENTS")
            if run.workflow is None or run.snapshot.workflow_id is None:
                raise FrontendError(503, "AUDIT_UNAVAILABLE")
            try:
                records = run.workflow.audit(run.snapshot.workflow_id)
            except KeyError:
                raise FrontendError(503, "AUDIT_UNAVAILABLE") from None
            selected = [event for event in records if event.sequence > after][:limit]
            summaries = []
            for event in selected:
                entry = event.agent_event
                tool = getattr(getattr(entry, "decision", None), "name", None)
                summaries.append(AuditSummary(sequence=event.sequence, timestamp=event.timestamp,
                    kind=event.kind, proposal_id=event.proposal_id, reviewer_user_id=event.reviewer_user_id,
                    proposal_status=event.proposal_status, tool=tool if tool in TOOLS else None,
                    error=safe_error(event.error or getattr(entry, "error", None))))
            cursor = summaries[-1].sequence if summaries else after
            return AuditPage(run_id=run_id, events=tuple(summaries), next_after=cursor,
                             has_more=bool(records and cursor < records[-1].sequence))
