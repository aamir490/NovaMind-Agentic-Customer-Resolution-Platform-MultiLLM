"""Validated domain records; no HTTP, AI, or storage dependencies."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
Subject = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CaseStatus(StrEnum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    ESCALATED = "escalated"


class Customer(Record):
    id: UUID
    name: Name


class OrderItem(Record):
    sku: Name
    name: Name
    quantity: int = Field(strict=True, ge=1, le=1000)


class Order(Record):
    id: UUID
    customer_id: UUID
    items: tuple[OrderItem, ...] = Field(min_length=1, max_length=100)


class SupportCase(Record):
    id: UUID
    customer_id: UUID
    order_id: UUID
    subject: Subject
    description: Description
    status: CaseStatus
    created_at: datetime
    updated_at: datetime


ALLOWED_TRANSITIONS = {
    CaseStatus.OPEN: frozenset({CaseStatus.IN_REVIEW, CaseStatus.ESCALATED}),
    CaseStatus.IN_REVIEW: frozenset({CaseStatus.ESCALATED}),
    CaseStatus.ESCALATED: frozenset(),
}


def can_transition(current: CaseStatus, target: CaseStatus) -> bool:
    """Repeated status requests are harmless; backward transitions are forbidden."""
    return target == current or target in ALLOWED_TRANSITIONS[current]
