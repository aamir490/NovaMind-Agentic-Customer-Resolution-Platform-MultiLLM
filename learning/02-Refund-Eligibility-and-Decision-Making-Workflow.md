## The short version

NovaMind is a controlled **case investigation and proposal system**, not an automated refund system.

For your NovaBook Laptop 10 example, the application can:

1. Record the customer’s order and complaint.
2. Start a refund-review workflow for a specific SKU.
3. Check trusted delivery evidence and deterministic demo-policy rules.
4. Create a pending refund or investigation proposal.
5. Pause for a reviewer/admin.
6. Record approval or rejection.

It **cannot** confirm what physically arrived, send money, arrange a return, ship a replacement, or update real inventory.

## Your “wrong NovaBook Laptop 10” example

### What actually happens

```text
Customer browses catalog
  → creates local/demo order for LAP-10
  → creates support case: “I received the wrong laptop”
  → staff starts a refund run with:
       resolution_intent = "refund"
       refund_sku = "LAP-10"
  → graph loads case and linked order
  → graph checks trusted delivery evidence for case + LAP-10
  → deterministic policy rules evaluate eligibility
  → creates PENDING_REVIEW proposal
  → LangGraph pauses
  → reviewer approves/rejects
  → only the proposal status changes
```

The customer’s written complaint is stored as case text. It is not automatically converted into verified facts such as “the delivered product was model X.”

The closest thing to a verified delivery record is `AuthoritativeDeliveryFact` in [operations.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\operations.py:90). It includes:

- case ID, order ID, customer ID
- purchased SKU and quantity
- delivery date
- reason, such as `wrong_item`
- item condition
- trusted source ID and source reference

It does **not** contain an “actual delivered SKU” field. So the system cannot compare “ordered NovaBook 10” against “delivered another laptop.” It trusts that a configured trusted source has already determined the delivery issue and marked it `wrong_item`.

For the Bedrock demo path, [bootstrap.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\container\bootstrap.py:196) seeds a synthetic LAP-10 case with `reason="wrong_item"` and a delivery date five days earlier. That is demo evidence, not a real carrier, warehouse, image, invoice, or commerce-system verification.

## How the complaint is investigated

### 1. Identity and ownership

The browser supplies a bearer token. The backend resolves it to a `CUSTOMER`, `REVIEWER`, or `ADMIN` identity using the local credential registry in [security.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\security.py:66).

A customer can only read or create records belonging to their bound `customer_id`. The backend checks ownership against live in-memory records; the client cannot choose another customer’s ID.

### 2. Case and order lookup

The agent graph always begins with the `get_case` tool in `_load_case()` in [graph_agent.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\graph_agent.py:108).

For a refund request, `_evaluate_refund()` then:

- fetches the linked order with `get_order`;
- verifies that the explicitly supplied `refund_sku` is in the order;
- runs `assess_refund_eligibility`;
- creates either a `refund` or `refund_investigation` proposal.

The approved tool list is defined in [tools.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\tools.py:89). The model cannot call a database, payment API, shell command, or arbitrary external service.

### 3. Does it verify that the complaint is genuine?

Only partially.

It can verify that:

- the case belongs to the customer;
- the order belongs to the same customer;
- the selected SKU appears in that order;
- a trusted delivery fact is present for the same case/order/customer/SKU;
- that trusted fact has internally consistent dates and quantities.

It cannot independently verify physical reality. There is no carrier API, warehouse system, proof-of-delivery feed, receipt parser, photo analysis, serial-number comparison, payment system, or fraud-detection service.

## How refund eligibility is decided

Refund eligibility is decided by deterministic Python—not by Bedrock or RAG.

`evaluate_rules()` in [operations.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\operations.py:144) rejects eligibility when:

- the SKU was not purchased;
- requested quantity exceeds purchased quantity;
- days since delivery exceeds the policy window;
- the reason is not allowed;
- the reason is `change_of_mind`, the policy requires unused condition, and the item is not unused.

The only current policy is a hardcoded demo `standard-return` policy in `BusinessOperations.__init__()`:

- 30-day return window;
- allowed reasons: `wrong_item`, `damaged`, `change_of_mind`;
- change-of-mind items must be unused;
- refund requires return;
- policy version: `demo-v1`.

The policy is code-held in memory, not stored in a policy database.

There are two eligibility modes:

- **General eligibility check**: takes staff-entered scenario inputs. Its result is explicitly based on unverified inputs and does not create trusted refund evidence.
- **Refund eligibility check**: requires a trusted delivery fact and creates evidence that a refund proposal can reference.

For missing trusted evidence, the system does not automatically reject the customer. It creates a `refund_investigation` proposal for human review. That behavior is in [graph_agent.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\graph_agent.py:177).

## Knowledge Base and RAG

The project has a small local knowledge corpus at [support.json](E:\codex-projetc\NovaMind-local-smoke-test\backend\knowledge\support.json).

It contains three demo documents:

- wrong-item/damaged-delivery intake guidance;
- return/refund guidance;
- proposal and human-review guidance.

The `search_knowledge` tool uses a local bag-of-words cosine match in [knowledge.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\knowledge.py:94). This is local lexical retrieval, not a managed RAG service, vector database, Bedrock Knowledge Base, or semantic embedding model.

Crucially, retrieved text is marked as reference-only. It cannot authorize a refund or override the deterministic rules. For explicit refund requests, the graph goes directly to deterministic evidence evaluation before model reasoning.

## What Amazon Bedrock / Nova actually decides

If configured, Amazon Nova Lite or Nova Pro receives:

- system instructions;
- the support-case context;
- optional conversation context;
- the list of allowed tools and their schemas;
- prior tool results.

The model may choose only:

- one approved tool call; or
- a `final` decision.

The Bedrock adapter in [bedrock.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\bedrock.py:150) makes a Converse call and requires structured JSON. For supported Nova models it uses forced formatting tools so the response must be a valid final/tool decision.

Nova does **not**:

- decide final refund eligibility;
- authenticate users;
- verify delivery evidence;
- identify reviewer identity;
- approve/reject a proposal;
- issue a refund;
- ship a replacement.

For an explicit refund, even tool planning is constrained: the graph itself chooses the order lookup, trusted-evidence assessment, and proposal type. The model is more relevant for non-refund investigations, where it can select read tools and possibly request a pending return/replacement proposal.

## What LangGraph does

LangGraph turns the workflow into named, controlled steps:

```text
load_case
  → evaluate_refund, if refund intent was explicit
  → otherwise reason
reason
  → execute_tool
  → record_result
  → reason again or finalize
proposal created
  → prepare_review
  → human_review interrupt
```

The graph has step limits and safe failure paths. It does not let the model jump directly to approval or execution. See [graph_agent.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\graph_agent.py:46) and [hitl.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\hitl.py:96).

## Human approval: what it means

Human approval is required because refund/replacement decisions are sensitive and the model is deliberately non-authoritative.

The reviewer receives:

- case ID and subject;
- workflow ID;
- proposal ID;
- review ID;
- proposed action: refund, refund investigation, return, or replacement;
- proposal rationale.

The reviewer decision must match the exact paused workflow/case/proposal/review identifiers. `HITLWorkflow._human_review()` verifies this and updates proposal status to `APPROVED` or `REJECTED`. See [hitl.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\hitl.py:212).

Approve means only:

> “The reviewer approved this recorded proposal.”

Reject means only:

> “The reviewer rejected this recorded proposal.”

Neither action triggers payment, stock movement, returns processing, email, shipping, or replacement fulfilment. The result explicitly says no action was executed.

## Scenario-by-scenario reality

- **Wrong product delivered**\
  Partially supported. `wrong_item` is a valid policy reason. A trusted source must provide the delivery fact; the system does not discover the wrong product itself.

- **Damaged laptop**\
  Partially supported. `damaged` is a valid reason and `damaged` is a valid item condition. It still needs trusted delivery evidence for the real refund path.

- **Defective product**\
  Not formally supported by current policy. `defective` is not one of the allowed `ReturnReason` values. The complaint can be recorded as free text, but the deterministic policy cannot evaluate it as a dedicated defect category.

- **Missing item**\
  Not formally supported. There is no missing-item reason, shipment-line model, partial-delivery record, or package reconciliation workflow.

- **Delayed delivery**\
  Not formally supported. There is no expected-delivery date, carrier tracking status, SLA rule, or compensation policy.

- **Refund outside the return window**\
  The rule returns `outside_return_window`. For an explicit refund workflow, the graph creates `refund_investigation` rather than automatically rejecting it. A human still decides.

- **Eligible refund**\
  Supported only when a trusted delivery fact exists, belongs to the same case/order/customer/SKU, and passes the demo policy. It creates a pending refund proposal; it does not issue money.

- **Missing evidence**\
  Supported as an investigation state. The refund workflow creates a `refund_investigation` proposal, not a refund approval.

- **Replacement request**\
  Partially supported. `replacement` exists as a proposal action. A configured model can request one in a non-refund workflow, then a reviewer can approve/reject it. There is no deterministic replacement policy, stock reservation, shipment, or actual replacement creation.

- **Partial refunds**\
  Not implemented. Proposals have no monetary amount, payment method, currency, line-item amount, or partial-refund action. Quantity is checked for eligibility but does not turn into a financial calculation.

## What must be added for an enterprise-grade platform

These are proposed capabilities, not claims about the current project.

### Business and integrations

- Real order-management, product-catalog, inventory, warehouse, shipping-carrier, payment/refund, CRM, notification, and return-label integrations.
- A fulfilment execution stage after approval, with idempotency, retries, reconciliation, and a final verified status.
- Monetary line items, taxes, discounts, payment references, refund amounts, partial refunds, exchanges, and replacement orders.
- Explicit business handling for defects, missing items, delivery delays, warranty claims, fraud, exceptions, and escalation.

### Data and policy

- A durable operational database for customers, orders, cases, delivery events, evidence, proposals, workflow state, and audit history.
- Versioned policies stored outside application code, with effective dates, regions, brands, product classes, exceptions, and decision explanations.
- A real evidence model containing actual delivered SKU, carrier events, warehouse picks, parcel IDs, proof of delivery, photos/documents, and verification provenance.
- A governed knowledge base with document ingestion, versioning, permissions, citations, retrieval evaluation, and policy/document distinction.

### AI and workflow

- Complaint-intent/entity extraction with confidence and human-confirmation paths.
- Domain-specific reasoning for replacement-versus-refund selection, grounded in inventory, policy, delivery evidence, and customer preferences.
- Model evaluation against realistic anonymized cases, safety tests, regressions, quality metrics, and monitored production behavior.
- Durable LangGraph checkpointing and long-running workflow recovery.
- Clear separation between AI recommendations, policy decisions, human authorization, and external execution results.

### Enterprise controls

- Real identity management such as Cognito/OIDC, user lifecycle, MFA/SSO, audit retention, permission scopes, and reviewer delegation.
- Durable observability, distributed tracing, business metrics, alerts, SLAs, and error queues.
- Privacy controls for customer data, secure evidence handling, retention/deletion policies, and compliance-grade audit logs.

For interviews, the strongest accurate description is: **NovaMind demonstrates a safe agentic decision-support architecture: constrained model planning, deterministic eligibility, evidence-bound refund proposals, and human approval—while deliberately stopping before real-world execution.**
