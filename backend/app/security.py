"""Local replaceable authentication and authorization, outside model/workflow data."""

from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, SecretStr, StringConstraints, model_validator

from .domain import InvestigationExecutionContext
from .service import CaseService, NotFoundError


class Role(StrEnum):
    CUSTOMER = "CUSTOMER"
    REVIEWER = "REVIEWER"
    ADMIN = "ADMIN"


class SecurityContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class AuthenticatedIdentity(SecurityContract):
    user_id: UUID
    role: Role
    customer_id: UUID | None = None

    @model_validator(mode="after")
    def customer_binding(self):
        if (self.role == Role.CUSTOMER) != (self.customer_id is not None):
            raise ValueError("Only CUSTOMER identities require a customer binding")
        return self


class LocalCredential(SecurityContract):
    token: SecretStr
    identity: AuthenticatedIdentity

    @model_validator(mode="after")
    def token_shape(self):
        token = self.token.get_secret_value()
        if not 16 <= len(token) <= 256 or not token.isascii() or any(c.isspace() for c in token):
            raise ValueError("Token must be 16-256 non-whitespace ASCII characters")
        return self


class ReviewInput(SecurityContract):
    # Identity is deliberately not a client field; unknown fields are rejected.
    note: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class SecurityError(Exception):
    def __init__(self, status_code=403):
        self.status_code = 401 if status_code == 401 else 403
        self.code = "UNAUTHENTICATED" if self.status_code == 401 else "FORBIDDEN"
        self.detail = "Authentication required" if self.status_code == 401 else "Access denied"
        super().__init__(self.detail)


class AuthenticationProvider(Protocol):
    def authenticate(self, token: str) -> AuthenticatedIdentity | None: ...


class LocalAuthenticationProvider:
    """An explicit server-side token registry. Empty by default; no public login."""

    def __init__(self, credentials: tuple[LocalCredential, ...] = ()):
        self._identities = {}
        subjects = {}
        for raw in credentials:
            credential = LocalCredential.model_validate(raw)
            identity = credential.identity
            digest = sha256(credential.token.get_secret_value().encode("ascii")).digest()
            if digest in self._identities or (identity.user_id in subjects and subjects[identity.user_id] != identity):
                raise ValueError("Duplicate token or conflicting identity")
            self._identities[digest] = identity
            subjects[identity.user_id] = identity
        self._subjects = subjects

    def authenticate(self, token: str) -> AuthenticatedIdentity | None:
        if not isinstance(token, str) or not 16 <= len(token) <= 256 or not token.isascii():
            return None
        return self._identities.get(sha256(token.encode("ascii")).digest())

    def identity_for_user_id(self, user_id: UUID) -> AuthenticatedIdentity | None:
        """Trusted bootstrap lookup; it never accepts a request credential."""
        if not isinstance(user_id, UUID):
            return None
        identity = self._subjects.get(user_id)
        return (None if identity is None
                else AuthenticatedIdentity.model_validate(identity))


_identity: ContextVar[AuthenticatedIdentity | None] = ContextVar("authenticated_identity", default=None)
_investigation_execution: ContextVar[InvestigationExecutionContext | None] = ContextVar(
    "investigation_execution", default=None,
)


@contextmanager
def authenticated(provider: AuthenticationProvider, token: str):
    """Trusted host entry point; never take a provider/identity from a model or request body."""
    try:
        identity = AuthenticatedIdentity.model_validate(provider.authenticate(token))
    except Exception:
        raise SecurityError(401) from None
    context_token = _identity.set(identity)
    try:
        yield identity
    finally:
        _identity.reset(context_token)


def current_identity() -> AuthenticatedIdentity:
    identity = _identity.get()
    if identity is None:
        raise SecurityError(401)
    return AuthenticatedIdentity.model_validate(identity)


def current_investigation_execution() -> InvestigationExecutionContext | None:
    """Return the active trusted job scope, if this is a server execution."""
    context = _investigation_execution.get()
    return None if context is None else InvestigationExecutionContext.model_validate(context)


@contextmanager
def trusted_investigation_execution(context: InvestigationExecutionContext):
    """Enter a server-only LangGraph tool scope for one marked job attempt.

    This cannot be combined with an HTTP-authenticated identity.  The caller
    must obtain ``context`` from the investigation-job repository after it has
    verified persisted claim, reservation, descriptor, and attempt records.
    """
    if _identity.get() is not None or _investigation_execution.get() is not None:
        raise RuntimeError("Investigation execution cannot be nested in a request identity")
    context = InvestigationExecutionContext.model_validate(context)
    scope_token = _investigation_execution.set(context)
    identity_token = _identity.set(AuthenticatedIdentity(
        user_id=context.worker_id, role=Role.REVIEWER,
    ))
    try:
        yield context
    finally:
        _identity.reset(identity_token)
        _investigation_execution.reset(scope_token)


def require_roles(*roles: Role) -> AuthenticatedIdentity:
    identity = current_identity()
    if identity.role not in roles:
        raise SecurityError()
    return identity


class Authorization:
    """Selectors are untrusted. Ownership always comes from live domain records."""

    def __init__(self, cases: CaseService, proposals=None):
        self._cases = cases
        self._proposals = proposals

    @staticmethod
    def customer(customer_id: UUID):
        execution = current_investigation_execution()
        if execution is not None and execution.customer_id != customer_id:
            raise SecurityError()
        identity = current_identity()
        if identity.role == Role.CUSTOMER and identity.customer_id != customer_id:
            raise SecurityError()

    def _owned(self, lookup, resource_id):
        identity = current_identity()
        try:
            record = lookup(resource_id)
        except NotFoundError:
            # Do not distinguish unknown IDs from another customer's IDs.
            if identity.role == Role.CUSTOMER or current_investigation_execution() is not None:
                raise SecurityError() from None
            raise
        self.customer(record.customer_id)
        return record

    def order(self, order_id: UUID):
        record = self._owned(self._cases.get_order, order_id)
        execution = current_investigation_execution()
        if execution is not None and record.id != execution.order_id:
            raise SecurityError()
        return record

    def case(self, case_id: UUID):
        record = self._owned(self._cases.get_case, case_id)
        execution = current_investigation_execution()
        if execution is not None and (
            record.id != execution.case_id or record.order_id != execution.order_id
        ):
            raise SecurityError()
        return record

    def proposal(self, proposal_id: UUID):
        identity = current_identity()
        try:
            record = self._proposals.get(proposal_id)
        except NotFoundError:
            if identity.role == Role.CUSTOMER or current_investigation_execution() is not None:
                raise SecurityError() from None
            raise
        self.case(record.case_id)
        return record

    def tool(self, name, request):
        current_identity()
        if name == "get_customer":
            self.customer(request.customer_id)
        elif name in ("get_case", "create_resolution_proposal"):
            self.case(request.case_id)
        elif name == "get_order":
            self.order(request.order_id)
        elif name == "assess_eligibility":
            self.order(request.order_id)
            if request.case_id is not None:
                self.case(request.case_id)
            self.customer(request.customer_id)
        elif name == "assess_refund_eligibility":
            self.case(request.case_id)
        elif name == "get_proposal_status":
            self.proposal(request.proposal_id)
