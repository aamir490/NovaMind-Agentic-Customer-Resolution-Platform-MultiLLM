## Result

NovaMind is a local/demo customer-support resolution platform. It lets customers create demo orders and support cases; lets staff investigate cases; and uses tightly bounded AI to propose—but never execute—refunds, returns, or replacements.

This was a **static, read-only source audit**. I did not run Docker, AWS, Bedrock, tests, or any external API, so runtime behavior is not newly verified.

## 1. Purpose and business problem

The project addresses the support-team problem of investigating complaints such as damaged, wrong, or unwanted purchases without letting an LLM make irreversible business decisions.

The core rule is:

> AI can investigate and propose; deterministic code checks rules; a human records approval or rejection; no financial or fulfilment action is performed.

The main executable composition is in [main.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\main.py:41). It creates services for cases, policies, inventory, proposals, local tools, authentication, telemetry, catalog, customer ordering, and the frontend workflow runtime.

## 2. Users

- **Customers** browse a demo catalog, add items to a browser-only cart, create their own local/demo orders, report problems, and view only their own cases/orders.
- **Reviewers** inspect cases, start and follow AI workflow runs, review pending proposals, approve/reject them, view workflow audit metadata, and inspect diagnostics.
- **Admins** can do everything reviewers can, plus use API endpoints to create customers and orders directly.
- Authentication is a server-side local bearer-token registry, not a real login/identity-provider flow. The default app has an empty registry, so all protected APIs deny access until configured. See [security.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\security.py:66) and [bootstrap.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\container\bootstrap.py:97).

## 3. What customers can currently use

The customer portal is a single React SPA view, not separate browser URLs. Its portal controller is [CustomerPortal.jsx](E:\codex-projetc\NovaMind-local-smoke-test\frontend\src\CustomerPortal.jsx:20).

Customers can:

- Browse/search/filter/sort a static seeded product catalog.
- Add catalog items to an in-browser cart.
- Create an order through `POST /api/my/orders`.
- List their own orders through `GET /api/my/orders`.
- Create a complaint/support request for one of their own orders through `POST /api/my/cases`.
- View their own cases through the ownership-filtered general case endpoints.
- See their case’s basic status: `open`, `in_review`, or `escalated`.

Important boundary: “placing an order” does not charge money, reserve/decrement inventory, or contact an external commerce platform. Catalog stock and prices are deterministic demo data in [catalog_seed.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\catalog_seed.py:1).

## 4. What reviewers/admins can currently use

The staff SPA has Dashboard, Cases, AI Workspace, Reviews, Audit, and Diagnostics views in [main.jsx](E:\codex-projetc\NovaMind-local-smoke-test\frontend\src\main.jsx:11).

They can:

- View support cases and their linked customer/order context.
- View static inventory and the applicable demo policy.
- Run a deterministic eligibility assessment using supplied scenario inputs.
- Change a case’s status through `PATCH /api/cases/{case_id}/status`.
- Start AI workflow runs, read their live events, and see pending review requests.
- Approve/reject a workflow-bound proposal.
- Read current-process workflow audit summaries and telemetry.

The backend, not the UI, enforces reviewer/admin permissions. For example, proposal approval routes require `REVIEWER` or `ADMIN` in [proposal_routes.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\proposal_routes.py:43).

## 5. AI capabilities actually implemented

Implemented in code:

- A structured-output LLM boundary: responses must validate as a strict `AgentDecision`, either one tool call or `final`. See [llm.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\llm.py:73).
- A bounded LangGraph workflow with a maximum number of reasoning steps, explicit nodes, and safe failure behavior. See [graph_agent.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\graph_agent.py:39).
- An allowlisted tool layer: case/customer/order lookups, policy/inventory lookup, eligibility checks, knowledge search, proposal creation, and proposal-status lookup. See [tools.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\tools.py:89).
- Local knowledge retrieval over `backend/knowledge/support.json`, using token/cosine matching—not a vector database and not semantic embeddings. See [knowledge.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\knowledge.py:94).
- Prompt-injection/payload guardrails and strict JSON validation. See [guardrails.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\guardrails.py:39).
- An Amazon Bedrock Converse adapter using `boto3`, bounded timeouts, one attempt, temperature `0`, and structured JSON handling. See [bedrock.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\bedrock.py:109).

Notably, the application-generated customer response is mostly a safe status message; it is not free-form model-written customer support prose.

## 6. How LangGraph, Bedrock, tools, and HITL fit together

```text
Authenticated user starts a case-bound run
  → FrontendRuntime creates an in-memory run
  → HITLWorkflow / LangGraph loads the case
  → refund request? deterministic refund-evidence branch
  → otherwise Bedrock/model chooses one allowlisted tool or “final”
  → application validates arguments and authorizes the tool
  → tool result returns to the graph
  → proposal created as PENDING_REVIEW, if appropriate
  → LangGraph interrupts for a reviewer/admin
  → reviewer/admin approves or rejects
  → proposal status is recorded; no real-world action occurs
```

`HITLWorkflow` uses LangGraph’s `interrupt()` at the human-review node, validates that the resume payload matches the exact workflow, case, proposal, and review IDs, then writes the human decision. See [hitl.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\hitl.py:212).

The model cannot provide reviewer identity, bypass the pause, call arbitrary code, or approve its own proposal.

## 7. Complaint and refund workflow

For an explicit refund request:

1. A customer creates an order and case, or an admin creates equivalent local records.
2. Staff starts a run with `resolution_intent: "refund"` and the exact `refund_sku`.
3. The graph loads the case, fetches its linked order, and verifies that SKU belongs to it.
4. `assess_refund_eligibility` requires a server-side trusted delivery fact—not customer-entered UI values.
5. The deterministic policy checks quantity, return window, allowed reason, and change-of-mind condition in `evaluate_rules()` in [operations.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\operations.py:144).
6. If trusted evidence supports eligibility, it creates a `refund` proposal. If evidence is absent/insufficient, it creates a `refund_investigation` proposal instead.
7. The workflow pauses at human review.
8. A reviewer/admin records `APPROVED` or `REJECTED`.
9. The workflow ends with “No action was executed.”

The refund-specific branch is deterministic before it reaches model reasoning; see `_evaluate_refund()` in [graph_agent.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\graph_agent.py:151).

## 8. API surface

Public:

- `GET /api/health`

Authenticated domain APIs:

- Customers: `POST /api/customers`, `GET /api/customers/{id}`
- Orders: `POST /api/orders`, `GET /api/orders/{id}`
- Cases: `POST/GET /api/cases`, `GET /api/cases/{id}`, `PATCH /api/cases/{id}/status`
- Operations: inventory, policy, general eligibility, and trusted refund eligibility
- Proposals: create/read/list/approve/reject
- Catalog: list, product by ID, product by SKU
- Customer-safe APIs: `POST/GET /api/my/orders`, `POST /api/my/cases`
- Workflow APIs: identity, runs, events, decisions, pending reviews, audit, diagnostics, optional conversations

The route implementations are in [routes.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\routes.py:23), [operation_routes.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\operation_routes.py:26), [frontend_routes.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\frontend_routes.py:46), and the customer route files.

## 9. Storage and integrations

Current running app state is primarily **in memory**:

- Cases/orders/customers: `CaseService`
- Proposals and review bindings: `ProposalService`
- Workflow runs/events/checkpoints: `FrontendRuntime` plus LangGraph `InMemorySaver`
- Catalog/inventory/policy: static deterministic seed data
- Trusted delivery facts/refund evidence: in-memory unless injected differently
- Telemetry: bounded in-process memory

SQLite repository, durable workflow-metadata, durable refund-evidence, and SQLite checkpoint classes do exist, but the standard `create_app()` wiring does not use them. [langgraph_checkpoints.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\langgraph_checkpoints.py:1) explicitly says runtime wiring is deferred.

External integration actually represented by executable code:

- Optional Amazon Bedrock runtime calls through `boto3`.
- No live CRM, OMS, payment gateway, shipping carrier, external inventory system, vector DB, S3 integration, or external database is wired into the application.

## 10. Implemented vs. partial/mocked/missing

Fully implemented as local application behavior:

- Authenticated role/ownership checks.
- Catalog/order/case creation and retrieval.
- Deterministic policy checks.
- Bounded tools, graph orchestration, pending proposals, HITL approval/rejection records.
- Frontend run/review/audit/diagnostic interfaces.
- Static local knowledge retrieval.

Present but configuration- or runtime-dependent:

- Bedrock: adapter and container configuration exist, but no live Bedrock call was made in this audit.
- AI runs: unavailable in default `create_app()` because no provider factory is configured; `/api/runs` returns `PROVIDER_UNAVAILABLE` in that condition. See [frontend_runtime.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\app\frontend_runtime.py:127).
- Conversation persistence: supported only when a SQLite conversation store is explicitly injected.
- Durable SQLite persistence/checkpoints: implemented/tested as components but not connected to the normal app factory.

Mocked/demo-only:

- Products, prices, stock, policies, customer records, and offline synthetic refund evidence.
- The local scripted provider always returns a fixed final decision; it is not intelligent model behavior. See [bootstrap.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\container\bootstrap.py:20).

Absent:

- Refund payment.
- Return labels/pickup/shipping.
- Replacement fulfilment.
- Inventory reservation/decrement.
- Card/payment processing.
- Durable default database.
- Real login/OIDC/Cognito.
- CI workflow or executable infrastructure-as-code.

## 11. Documentation mismatch worth knowing

The current source supports **Amazon Bedrock** as its only cloud LLM provider. [bootstrap.py](E:\codex-projetc\NovaMind-local-smoke-test\backend\container\bootstrap.py:131) selects only explicit Bedrock configuration; without it, cloud model runs remain unavailable.

Therefore, README and notes should be read against the current Bedrock-only executable runtime. The actual `infrastructure/` directory contains planning documentation only, not AWS definitions.

## 12. Tests and what they verify

Static source count: **507 Python test methods** across 36 files and **138 Node frontend tests** across 11 files. I did not execute them.

They cover:

- Domain records, status transitions, HTTP routes, authorization, customer ownership, and validation.
- Deterministic policy/refund-evidence behavior.
- Tool allowlisting, malformed model output, guardrails, knowledge retrieval, and agent limits.
- LangGraph flow, interruption/resume matching, duplicate decision protection, and proposal binding.
- Bedrock request/response parsing using mocked clients—not live AWS.
- SQLite component behavior using temporary test databases.
- Container bootstrap/configuration contracts without building/running Docker.
- Frontend API-contract validation, catalog/cart behavior, customer intake, staff views, audit/diagnostics rendering, and theme behavior.
- Phase 15’s deterministic offline evaluation harness: 15 tests exercising 54 scripted scenarios.

Examples: [test_refund_orchestration_m3a.py](E:\codex-projetc\NovaMind-local-smoke-test\tests\test_refund_orchestration_m3a.py), [test_hitl.py](E:\codex-projetc\NovaMind-local-smoke-test\tests\test_hitl.py), and [frontend-customer-portal.test.mjs](E:\codex-projetc\NovaMind-local-smoke-test\tests\frontend-customer-portal.test.mjs).

These are strong code-level tests, but they do **not** newly prove a running Docker stack, live Bedrock credentials/model access, a deployed AWS environment, or real refund execution.

## 13. Three realistic supported scenarios

1. **Customer reports a wrong item**
   - Customer selects demo products, creates an order, then submits “Received the wrong item.”
   - The system creates an `open` support case owned by that customer.
   - Staff can inspect the linked order and use the deterministic policy assessment.

2. **Staff requests a refund review for a verified delivery**
   - A trusted host has supplied a delivery fact for the exact case/order/SKU.
   - Staff starts a refund-intent run with that SKU.
   - The graph verifies evidence, creates a pending refund proposal, and requires a reviewer decision.
   - Approval records `APPROVED`; it does not issue money.

3. **Staff investigates a general complaint**
   - Staff opens a case, order, catalog/inventory and policy data.
   - With a configured Bedrock provider, the bounded agent may request permitted read tools or create one pending return/replacement proposal.
   - The workflow either finishes informationally or pauses for human review.

No files were created or modified. I also did not create a Canvas artifact because your strict read-only rule prohibits creating the required `.canvas.tsx` file.
