"""Read-only local business operations using an explicit demonstration policy."""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

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
    assessment_basis: Literal["local_scenario_unverified_inputs"] = "local_scenario_unverified_inputs"
    authorization_granted: Literal[False] = False


class EligibilityInput(Record):
    customer_id: UUID
    sku: Name
    quantity: int = Field(strict=True, ge=1, le=1000)
    days_since_delivery: int = Field(strict=True, ge=0, le=36500)
    reason: ReturnReason
    condition: ItemCondition


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
            "LAP-1": InventoryItem(sku="LAP-1", available_quantity=5, policy_id="standard-return"),
            "LAP-2": InventoryItem(sku="LAP-2", available_quantity=0, policy_id="standard-return"),
        }

    def get_inventory(self, sku: str) -> InventoryItem:
        if sku not in self._inventory:
            raise NotFoundError("Inventory item not found")
        return self._inventory[sku]

    def get_policy(self, policy_id: str) -> ReturnPolicy:
        if policy_id not in self._policies:
            raise NotFoundError("Policy not found")
        return self._policies[policy_id]

    @observed("eligibility")
    def check_eligibility(self, order_id: UUID, request: EligibilityInput) -> EligibilityResult:
        self._cases.get_customer(request.customer_id)
        order = self._cases.get_order(order_id)
        if order.customer_id != request.customer_id:
            raise ConflictError("Order does not belong to the selected customer")
        inventory = self.get_inventory(request.sku)
        policy = self.get_policy(inventory.policy_id)
        purchased = sum(item.quantity for item in order.items if item.sku == request.sku)
        reasons = evaluate_rules(policy, request, purchased)
        return EligibilityResult(
            order_id=order.id, customer_id=request.customer_id, sku=request.sku,
            requested_quantity=request.quantity, purchased_quantity=purchased,
            days_since_delivery=request.days_since_delivery,
            reason=request.reason, condition=request.condition,
            policy_id=policy.id, policy_version=policy.version,
            return_eligible=not reasons, refund_eligible=not reasons,
            denial_reasons=reasons, refund_requires_return=policy.refund_requires_return,
        )
