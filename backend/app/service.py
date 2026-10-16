"""Local application service with process-local storage and atomic mutations."""

from datetime import datetime, timezone
from threading import RLock
from typing import Protocol
from uuid import UUID, uuid4

from .domain import Customer, Order, SupportCase, CaseStatus, can_transition
from .schemas import CustomerCreate, OrderCreate, CaseCreate


class NotFoundError(Exception):
    pass


class ConflictError(Exception):
    pass


class CaseRepository(Protocol):
    """Durable business-record boundary used only when explicitly configured."""

    def seed_customer(self, customer: Customer) -> None: ...
    def create_customer(self, customer: Customer) -> Customer: ...
    def get_customer(self, customer_id: UUID) -> Customer: ...
    def create_order(self, order: Order) -> Order: ...
    def get_order(self, order_id: UUID) -> Order: ...
    def create_case(self, case: SupportCase) -> SupportCase: ...
    def get_case(self, case_id: UUID) -> SupportCase: ...
    def list_orders(self, customer_id: UUID | None = None) -> list[Order]: ...
    def list_cases(self, customer_id: UUID | None = None) -> list[SupportCase]: ...
    def update_case(self, case: SupportCase) -> SupportCase: ...


class CaseService:
    """Business records backed by memory by default, or an explicit repository."""

    def __init__(self, repository: CaseRepository | None = None) -> None:
        self._customers: dict[UUID, Customer] = {}
        self._orders: dict[UUID, Order] = {}
        self._cases: dict[UUID, SupportCase] = {}
        self._repository = repository
        self._lock = RLock()

    def seed_customer(self, customer: Customer) -> None:
        """
        Insert a pre-constructed Customer record at application startup.

        Intended exclusively for deterministic demo / local-dev initialisation
        driven by trusted startup code (create_app, container bootstrap).
        Must never be called from an HTTP route handler or any path reachable
        by external request data.

        Duplicate handling:
        - Same id, same record  → silent no-op (idempotent restart).
        - Same id, different record → raises ValueError to prevent silent
          corruption of an already-seeded identity.
        """
        with self._lock:
            if self._repository is not None:
                self._repository.seed_customer(customer)
                return
            existing = self._customers.get(customer.id)
            if existing is not None:
                if existing != customer:
                    raise ValueError(
                        f"Startup seed conflict: customer {customer.id} already exists "
                        "with different data"
                    )
                return  # idempotent no-op
            self._customers[customer.id] = customer

    def create_customer(self, request: CustomerCreate) -> Customer:
        with self._lock:
            customer = Customer(id=uuid4(), name=request.name)
            if self._repository is not None:
                return self._repository.create_customer(customer)
            self._customers[customer.id] = customer
            return customer

    def get_customer(self, customer_id: UUID) -> Customer:
        with self._lock:
            if self._repository is not None:
                return self._repository.get_customer(customer_id)
            if customer_id not in self._customers:
                raise NotFoundError("Customer not found")
            return self._customers[customer_id]

    def create_order(self, request: OrderCreate) -> Order:
        with self._lock:
            self.get_customer(request.customer_id)
            order = Order(id=uuid4(), **request.model_dump())
            if self._repository is not None:
                return self._repository.create_order(order)
            self._orders[order.id] = order
            return order

    def get_order(self, order_id: UUID) -> Order:
        with self._lock:
            if self._repository is not None:
                return self._repository.get_order(order_id)
            if order_id not in self._orders:
                raise NotFoundError("Order not found")
            return self._orders[order_id]

    def create_case(self, request: CaseCreate) -> SupportCase:
        with self._lock:
            self.get_customer(request.customer_id)
            order = self.get_order(request.order_id)
            if order.customer_id != request.customer_id:
                raise ConflictError("Order does not belong to the selected customer")
            now = datetime.now(timezone.utc)
            case = SupportCase(
                id=uuid4(), status=CaseStatus.OPEN,
                created_at=now, updated_at=now, **request.model_dump(),
            )
            if self._repository is not None:
                return self._repository.create_case(case)
            self._cases[case.id] = case
            return case

    def get_case(self, case_id: UUID) -> SupportCase:
        with self._lock:
            if self._repository is not None:
                return self._repository.get_case(case_id)
            if case_id not in self._cases:
                raise NotFoundError("Case not found")
            return self._cases[case_id]

    def list_orders(self, customer_id: UUID | None = None) -> list[Order]:
        with self._lock:
            if customer_id is not None:
                self.get_customer(customer_id)
            if self._repository is not None:
                return self._repository.list_orders(customer_id)
            return [order for order in self._orders.values()
                    if customer_id is None or order.customer_id == customer_id]

    def list_cases(self, customer_id: UUID | None = None) -> list[SupportCase]:
        with self._lock:
            if customer_id is not None:
                self.get_customer(customer_id)
            if self._repository is not None:
                return self._repository.list_cases(customer_id)
            return [case for case in self._cases.values()
                    if customer_id is None or case.customer_id == customer_id]

    def update_status(self, case_id: UUID, status: CaseStatus) -> SupportCase:
        with self._lock:
            case = self.get_case(case_id)
            if not can_transition(case.status, status):
                raise ConflictError(f"Cannot change case status from {case.status} to {status}")
            if case.status == status:
                return case
            updated = SupportCase(**{
                **case.model_dump(), "status": status,
                "updated_at": datetime.now(timezone.utc),
            })
            if self._repository is not None:
                return self._repository.update_case(updated)
            self._cases[case_id] = updated
            return updated
