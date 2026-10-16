"""Read-only local business operations using an explicit demonstration policy."""

from datetime import datetime, timezone
from enum import StrEnum
from threading import RLock
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field

from .catalog_seed import PRODUCT_SEEDS
from .domain import Name, Record
from .service import CaseService, ConflictError, NotFoundError
from .observability import observed


class ReturnReason(StrEnum):
    WRONG_ITEM = "wrong_item"
    DAMAGED = "damaged"
    CHANGE_OF_MIND = "change_of_mind"


class ItemCondition(StrEnum):
    UNUSED = "unused"
    USED = "used"
    DAMAGED = "damaged"


class ReturnPolicy(Record):
    id: Name
    version: Name
    return_window_days: int = Field(ge=0)
    allowed_reasons: tuple[ReturnReason, ...]
    change_of_mind_requires_unused: bool
    refund_requires_return: bool
    description: str


class InventoryItem(Record):
    sku: Name
    available_quantity: int = Field(ge=0)
    policy_id: Name


class EligibilityResult(Record):
    evidence_id: UUID | None = None
    case_id: UUID | None = None
    order_id: UUID
    customer_id: UUID
    sku: Name
    requested_quantity: int
    purchased_quantity: int
    days_since_delivery: int
    reason: ReturnReason
    condition: ItemCondition
    policy_id: Name
    policy_version: Name
    return_eligible: bool
    refund_eligible: bool
    denial_reasons: tuple[str, ...]
    refund_requires_return: bool
    assessment_basis: Literal["local_scenario_unverified_inputs", "trusted_delivery_record"] = "local_scenario_unverified_inputs"
    authorization_granted: Literal[False] = False


class EligibilityInput(Record):
    customer_id: UUID
    sku: Name
    quantity: int = Field(strict=True, ge=1, le=1000)
    days_since_delivery: int = Field(strict=True, ge=0, le=36500)
    reason: ReturnReason
    condition: ItemCondition


class EligibilityEvidence(Record):
    """Server-issued, immutable evidence for one case/order/item assessment."""

    id: UUID
    case_id: UUID
    order_id: UUID
    customer_id: UUID
    sku: Name
    result: EligibilityResult
    created_at: datetime


class DeliveryVerificationStatus(StrEnum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"


class AuthoritativeDeliveryFact(Record):
    """A trusted-host delivery/return verification record; never accepted from HTTP or tools."""

    id: UUID
    case_id: UUID
    order_id: UUID
    customer_id: UUID
    sku: Name
    quantity: int = Field(strict=True, ge=1, le=1000)
    delivered_at: datetime
    reason: ReturnReason
    condition: ItemCondition
    source_reference: Name
    verification_status: DeliveryVerificationStatus
    verified_at: datetime | None = None


def evaluate_rules(policy: ReturnPolicy, request: EligibilityInput,
                   purchased_quantity: int) -> tuple[str, ...]:
    """Pure rule evaluation. No clock, I/O, or record mutation."""
    reasons = []
    if purchased_quantity == 0:
        reasons.append("item_not_in_order")
    elif request.quantity > purchased_quantity:
        reasons.append("quantity_exceeds_purchased")
    if request.days_since_delivery > policy.return_window_days:
        reasons.append("outside_return_window")
    if request.reason not in policy.allowed_reasons:
        reasons.append("reason_not_allowed")
    if (request.reason == ReturnReason.CHANGE_OF_MIND
            and policy.change_of_mind_requires_unused
            and request.condition != ItemCondition.UNUSED):
        reasons.append("change_of_mind_requires_unused")
    return tuple(reasons)


class BusinessOperations:
    def __init__(self, cases: CaseService) -> None:
        self._cases = cases
        self._policies = {
            "standard-return": ReturnPolicy(
                id="standard-return", version="demo-v1", return_window_days=30,
                allowed_reasons=tuple(ReturnReason),
                change_of_mind_requires_unused=True, refund_requires_return=True,
                description=("Demo policy, not real merchant terms. Returns within 30 days; "
                             "change of mind requires unused condition. Refund eligibility "
                             "is conditional on a verified return and later authorization."),
            ),
        }
        self._inventory = {
            product.sku: InventoryItem(
                sku=product.sku,
                available_quantity=product.stock_quantity,
                policy_id="standard-return",
            )
            for product in PRODUCT_SEEDS
        }
        self._eligibility_evidence: dict[UUID, EligibilityEvidence] = {}
        self._delivery_facts: dict[tuple[UUID, str], AuthoritativeDeliveryFact] = {}
        self._evidence_lock = RLock()

    def get_inventory(self, sku: str) -> InventoryItem:
        if sku not in self._inventory:
            raise NotFoundError("Inventory item not found")
        return self._inventory[sku]

    def get_policy(self, policy_id: str) -> ReturnPolicy:
        if policy_id not in self._policies:
            raise NotFoundError("Policy not found")
        return self._policies[policy_id]

    @observed("eligibility")
    def check_eligibility(self, order_id: UUID, request: EligibilityInput,
                          *, case_id: UUID | None = None) -> EligibilityResult:
        self._cases.get_customer(request.customer_id)
        order = self._cases.get_order(order_id)
        if order.customer_id != request.customer_id:
            raise ConflictError("Order does not belong to the selected customer")
        if case_id is not None:
            case = self._cases.get_case(case_id)
            if case.customer_id != request.customer_id or case.order_id != order.id:
                raise ConflictError("Case does not belong to the selected customer and order")
        inventory = self.get_inventory(request.sku)
        policy = self.get_policy(inventory.policy_id)
        purchased = sum(item.quantity for item in order.items if item.sku == request.sku)
        reasons = evaluate_rules(policy, request, purchased)
        result = EligibilityResult(
            case_id=case_id,
            order_id=order.id, customer_id=request.customer_id, sku=request.sku,
            requested_quantity=request.quantity, purchased_quantity=purchased,
            days_since_delivery=request.days_since_delivery,
            reason=request.reason, condition=request.condition,
            policy_id=policy.id, policy_version=policy.version,
            return_eligible=not reasons, refund_eligible=not reasons,
            denial_reasons=reasons, refund_requires_return=policy.refund_requires_return,
        )
        return result

    def record_authoritative_delivery_fact(self, fact: AuthoritativeDeliveryFact) -> None:
        """Trusted-host integration boundary. Deliberately has no HTTP or tool exposure."""
        fact = AuthoritativeDeliveryFact.model_validate(fact.model_dump())
        case = self._cases.get_case(fact.case_id)
        order = self._cases.get_order(fact.order_id)
        if (case.order_id != order.id or case.customer_id != fact.customer_id
                or order.customer_id != fact.customer_id):
            raise ConflictError("Delivery fact does not match the current case, order, and customer")
        purchased = sum(item.quantity for item in order.items if item.sku == fact.sku)
        if purchased == 0 or fact.quantity > purchased:
            raise ConflictError("Delivery fact does not match an order item")
        if fact.delivered_at.tzinfo is None or (fact.verified_at is not None and fact.verified_at.tzinfo is None):
            raise ConflictError("Delivery fact timestamps must include a timezone")
        if fact.verification_status == DeliveryVerificationStatus.VERIFIED and fact.verified_at is None:
            raise ConflictError("Verified delivery facts require a verification timestamp")
        if fact.verified_at is not None and fact.verified_at < fact.delivered_at:
            raise ConflictError("Delivery fact verification cannot predate delivery")
        key = (fact.case_id, fact.sku)
        with self._evidence_lock:
            if key in self._delivery_facts:
                raise ConflictError("Authoritative delivery fact already exists for this case item")
            self._delivery_facts[key] = fact

    @observed("refund_eligibility")
    def assess_refund_eligibility(self, case_id: UUID, sku: str) -> EligibilityResult:
        case = self._cases.get_case(case_id)
        key = (case.id, sku)
        with self._evidence_lock:
            fact = self._delivery_facts.get(key)
        if fact is None:
            raise ConflictError("Trusted refund evidence unavailable: delivery facts are missing")
        if fact.verification_status != DeliveryVerificationStatus.VERIFIED or fact.verified_at is None:
            raise ConflictError("Trusted refund evidence unavailable: delivery facts are unverified")
        order = self._cases.get_order(case.order_id)
        if (fact.order_id != order.id or fact.customer_id != case.customer_id
                or order.customer_id != case.customer_id):
            raise ConflictError("Trusted refund evidence does not match the current case and order")
        inventory = self.get_inventory(fact.sku)
        policy = self.get_policy(inventory.policy_id)
        now = datetime.now(timezone.utc)
        delivered_at = fact.delivered_at.astimezone(timezone.utc)
        if delivered_at > now:
            raise ConflictError("Trusted refund evidence unavailable: delivery timestamp is in the future")
        days_since_delivery = (now.date() - delivered_at.date()).days
        purchased = sum(item.quantity for item in order.items if item.sku == fact.sku)
        assessment = EligibilityInput(
            customer_id=case.customer_id, sku=fact.sku, quantity=fact.quantity,
            days_since_delivery=days_since_delivery, reason=fact.reason, condition=fact.condition,
        )
        reasons = evaluate_rules(policy, assessment, purchased)
        evidence_id = uuid4()
        result = EligibilityResult(
            evidence_id=evidence_id, case_id=case.id, order_id=order.id,
            customer_id=case.customer_id, sku=fact.sku, requested_quantity=fact.quantity,
            purchased_quantity=purchased, days_since_delivery=days_since_delivery,
            reason=fact.reason, condition=fact.condition, policy_id=policy.id,
            policy_version=policy.version, return_eligible=not reasons,
            refund_eligible=not reasons, denial_reasons=reasons,
            refund_requires_return=policy.refund_requires_return,
            assessment_basis="trusted_delivery_record",
        )
        evidence = EligibilityEvidence(
            id=evidence_id, case_id=case.id, order_id=order.id,
            customer_id=case.customer_id, sku=fact.sku, result=result,
            created_at=now,
        )
        with self._evidence_lock:
            self._eligibility_evidence[evidence.id] = evidence
        return result

    def get_eligibility_evidence(self, evidence_id: UUID) -> EligibilityEvidence:
        with self._evidence_lock:
            evidence = self._eligibility_evidence.get(evidence_id)
        if evidence is None:
            raise NotFoundError("Eligibility evidence not found")
        return evidence
