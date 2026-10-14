"""Thin local HTTP routes; relationships and transitions live in the service."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request

from .domain import Customer, Order, SupportCase
from .schemas import CustomerCreate, OrderCreate, CaseCreate, CaseStatusUpdate
from .service import CaseService
from .security import Authorization, Role, current_identity, require_roles

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


@router.patch("/cases/{case_id}/status", response_model=SupportCase, tags=["cases"])
def update_status(case_id: UUID, body: CaseStatusUpdate, service: Service) -> SupportCase:
    require_roles(Role.REVIEWER, Role.ADMIN)
    Authorization(service).case(case_id)
    return service.update_status(case_id, body.status)
