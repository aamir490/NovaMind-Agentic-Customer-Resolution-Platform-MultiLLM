"""Grounded, non-executing resolution drafts for supported complaint types."""

import re
from enum import StrEnum
from typing import Literal

from pydantic import Field

from .domain import Order, Record, SupportCase


class IssueType(StrEnum):
    DAMAGED = "damaged"
    WRONG_ITEM = "wrong_item"
    MISSING_ITEM = "missing_item"
    DELAYED_DELIVERY = "delayed_delivery"
    MALFUNCTION = "malfunction"
    RETURN = "return"
    REFUND_NOT_RECEIVED = "refund_not_received"


class RecommendedDecision(StrEnum):
    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"
    ESCALATE = "ESCALATE"


class CustomerIntent(StrEnum):
    REPORT_ISSUE = "report_issue"
    TRACK_DELIVERY = "track_delivery"
    REQUEST_RETURN = "request_return"
    REQUEST_REPLACEMENT = "request_replacement"
    REQUEST_REFUND = "request_refund"


InvestigationTool = Literal[
    "get_customer", "get_order", "get_case", "get_inventory", "get_policy", "search_knowledge",
    "assess_eligibility", "assess_refund_eligibility", "create_resolution_proposal", "get_proposal_status",
]


class InvestigationPlan(Record):
    """Application-owned, read-only investigation boundary for one case."""

    issue_type: IssueType
    customer_intent: CustomerIntent
    allowed_tools: tuple[InvestigationTool, ...] = Field(min_length=1, max_length=10)
    required_evidence: tuple[str, ...] = Field(min_length=1, max_length=6)
    max_tool_calls: int = Field(strict=True, ge=1, le=32)
    actions_executed: Literal[False] = False


class ResolutionDraft(Record):
    """Application-rendered draft; this is never an approval or execution."""

    issue_type: IssueType
    recommended_decision: RecommendedDecision
    verified_facts: tuple[str, ...] = Field(min_length=1, max_length=8)
    missing_evidence: tuple[str, ...] = Field(min_length=1, max_length=6)
    decision_reason: str = Field(min_length=1, max_length=1200)
    customer_facing_response: str = Field(min_length=1, max_length=1200)
    actions_executed: Literal[False] = False


def classify_issue_type(*, subject: str, description: str, message: str) -> IssueType:
    """Classify only from case/request text; ambiguous input defaults conservatively."""
    text = " ".join((subject, description, message)).casefold()
    if re.search(r"\b(refund|money|payment).{0,40}\b(not received|missing|hasn.t arrived|never arrived)\b", text):
        return IssueType.REFUND_NOT_RECEIVED
    if re.search(r"\bwrong (item|product|order)|incorrect (item|product)|different item\b", text):
        return IssueType.WRONG_ITEM
    if re.search(r"\bmissing (item|product|part)|item (is )?missing|not in (the )?(box|package)\b", text):
        return IssueType.MISSING_ITEM
    if re.search(r"\b(delayed|late|overdue|not delivered|delivery delay)\b", text):
        return IssueType.DELAYED_DELIVERY
    if re.search(
        r"\b(malfunction|defect|defective|not working|stopped working|broken|"
        r"not functioning(?: properly)?|(?:doesn't|does not|won't|will not) work|"
        r"unresponsive|fails? to (?:work|start|operate)|"
        r"(?:storage|device|product|item)(?:\s+\w+){0,3}\s+(?:is\s+)?not detected|"
        r"(?:storage|device|product|item)(?:\s+\w+){0,3}\s+(?:cannot|can't) be detected)\b",
        text,
    ):
        return IssueType.MALFUNCTION
    if re.search(r"\b(damaged|damage|cracked|scratched|broken on arrival)\b", text):
        return IssueType.DAMAGED
    return IssueType.RETURN


def build_resolution_draft(case: SupportCase, order: Order, *,
                           trusted_delivery_evidence: bool,
                           policy_id: str, policy_version: str,
                           message: str) -> ResolutionDraft:
    """Create a bounded draft from records and evidence availability only.

    It intentionally does not infer an item, delivery date, payment result, or
    eligibility.  Policy/evidence absence is a reason to request records or
    escalate, never a reason to approve a refund or replacement.
    """
    issue = classify_issue_type(
        subject=case.subject, description=case.description, message=message,
    )
    facts = [f"case_id:{case.id}", f"order_id:{order.id}"]
    if order.items:
        facts.append("ordered_skus:" + ",".join(sorted(item.sku for item in order.items)))
    if issue in (IssueType.RETURN, IssueType.REFUND_NOT_RECEIVED):
        facts.append(f"policy:{policy_id}@{policy_version}")
    if trusted_delivery_evidence:
        facts.append("trusted_delivery_evidence:present")

    if issue is IssueType.REFUND_NOT_RECEIVED:
        decision = RecommendedDecision.ESCALATE
        missing = ("trusted payment/refund settlement record",)
        reason = "Payment or refund settlement evidence is not available in the verified case records."
    elif issue is IssueType.DELAYED_DELIVERY:
        decision = RecommendedDecision.ESCALATE
        missing = ("trusted shipment tracking status", "committed delivery date")
        reason = "Delivery timing cannot be established from the available verified records."
    elif issue is IssueType.RETURN:
        decision = RecommendedDecision.REQUEST_EVIDENCE
        missing = (("affected item and condition", "return or delivery evidence")
                   if trusted_delivery_evidence else
                   ("affected item and condition", "trusted shipping or delivery evidence"))
        reason = "The policy is available, but required item and condition evidence is not verified."
    elif not trusted_delivery_evidence:
        decision = RecommendedDecision.REQUEST_EVIDENCE
        missing = ("trusted shipping or delivery evidence", "case-specific supporting evidence")
        reason = "The complaint is recorded, but shipment/delivery facts have not been verified."
    else:
        decision = RecommendedDecision.ESCALATE
        missing = ("staff review of the case-specific supporting evidence",)
        reason = "A delivery record exists, but the issue still requires a staff decision under policy."

    response = (
        "We have recorded your request. "
        + ("We need the listed evidence before eligibility can be assessed. "
           if decision is RecommendedDecision.REQUEST_EVIDENCE
           else "A staff member must review the available records and missing evidence. ")
        + "No refund, replacement, return, or payment action has been executed."
    )
    return ResolutionDraft(
        issue_type=issue, recommended_decision=decision, verified_facts=tuple(facts),
        missing_evidence=missing, decision_reason=reason,
        customer_facing_response=response,
    )


def build_investigation_plan(draft: ResolutionDraft, *, request_text: str,
                             refund_requested: bool, max_tool_calls: int) -> InvestigationPlan:
    """Return a small issue-specific allowlist without authorizing any action.

    Refund intent is trusted only when the request schema supplied it explicitly;
    customer prose alone cannot put the refund assessment tool in an LLM plan.
    """
    issue = draft.issue_type
    if refund_requested:
        intent = CustomerIntent.REQUEST_REFUND
        tools: tuple[InvestigationTool, ...] = (
            "get_case", "get_order", "assess_refund_eligibility", "create_resolution_proposal", "get_proposal_status",
        )
    elif issue is IssueType.DELAYED_DELIVERY:
        intent = CustomerIntent.TRACK_DELIVERY
        tools = ("get_case", "get_order", "search_knowledge", "create_resolution_proposal", "get_proposal_status")
    elif issue is IssueType.REFUND_NOT_RECEIVED:
        intent = CustomerIntent.REPORT_ISSUE
        tools = ("get_case", "get_order", "search_knowledge", "create_resolution_proposal", "get_proposal_status")
    elif issue is IssueType.RETURN:
        intent = CustomerIntent.REQUEST_RETURN
        tools = ("get_case", "get_order", "get_policy", "search_knowledge", "create_resolution_proposal", "get_proposal_status")
    elif issue in (IssueType.DAMAGED, IssueType.WRONG_ITEM):
        intent = CustomerIntent.REQUEST_REPLACEMENT if re.search(
            r"\b(replace|replacement|exchange)\b", request_text, re.IGNORECASE,
        ) else CustomerIntent.REPORT_ISSUE
        tools = (
            "get_customer", "get_case", "get_order", "get_inventory", "get_policy", "assess_eligibility",
            "search_knowledge", "create_resolution_proposal", "get_proposal_status",
        )
    elif issue is IssueType.MISSING_ITEM:
        # This issue type has no authoritative ReturnReason.  Do not coerce it
        # into another policy reason; collect evidence and escalate instead.
        intent = CustomerIntent.REQUEST_REPLACEMENT if re.search(
            r"\b(replace|replacement|exchange)\b", request_text, re.IGNORECASE,
        ) else CustomerIntent.REPORT_ISSUE
        tools = (
            "get_customer", "get_case", "get_order", "get_inventory", "get_policy",
            "search_knowledge", "create_resolution_proposal", "get_proposal_status",
        )
    elif issue is IssueType.MALFUNCTION:
        intent = CustomerIntent.REQUEST_REPLACEMENT if re.search(
            r"\b(replace|replacement|exchange)\b", request_text, re.IGNORECASE,
        ) else CustomerIntent.REPORT_ISSUE
        tools = (
            "get_customer", "get_case", "get_order", "get_inventory",
            "search_knowledge", "create_resolution_proposal", "get_proposal_status",
        )
    else:  # Defensive fallback if a future issue type is introduced without a plan.
        intent = CustomerIntent.REPORT_ISSUE
        tools = ("get_case", "get_order", "search_knowledge", "create_resolution_proposal", "get_proposal_status")
    return InvestigationPlan(
        issue_type=issue, customer_intent=intent, allowed_tools=tools,
        required_evidence=draft.missing_evidence, max_tool_calls=max_tool_calls,
    )
