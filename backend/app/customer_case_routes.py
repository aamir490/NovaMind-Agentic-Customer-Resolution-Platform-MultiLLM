"""
Customer-safe support case intake.

POST /api/my/cases  — create a SupportCase for the authenticated customer

Security contract
-----------------
- customer_id is ALWAYS derived from the authenticated identity; it is never
  accepted from the request body (extra="forbid" on CustomerCaseCreate).
- Requires Role.CUSTOMER.
- Order ownership is verified through the existing Authorization mechanism
  before CaseService is called.
- The existing POST /api/cases (admin/staff) route is untouched.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from .domain import SupportCase
from .routes import Service
from .schemas import CaseCreate, CustomerCaseCreate
from .security import Authorization, Role, require_roles

router = APIRouter(prefix="/api/my", tags=["customer-cases"])


@router.post("/cases", response_model=SupportCase, status_code=201)
def create_my_case(body: CustomerCaseCreate, service: Service) -> SupportCase:
    """
    Create a support case for the authenticated customer.

    The customer_id is derived exclusively from the server-side authenticated
    identity and must NOT be supplied in the request body — extra fields are
    rejected (422).

    The order must belong to the authenticated customer; an order owned by
    another customer is indistinguishable from a missing order (403).
    """
    identity = require_roles(Role.CUSTOMER)

    # customer_id comes from the server-side identity, never from the request.
    customer_id = identity.customer_id

    # Verify the order exists and belongs to this customer.
    # Authorization._owned() raises SecurityError(403) for cross-customer or
    # missing orders when the caller is Role.CUSTOMER.
    Authorization(service).order(body.order_id)

    # Build the internal CaseCreate using the server-derived customer_id.
    case_input = CaseCreate(
        customer_id=customer_id,
        order_id=body.order_id,
        subject=body.subject,
        description=body.description,
    )
    return service.create_case(case_input)
