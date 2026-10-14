# 00 — Project overview

This chapter preserves the Phase 0 learning snapshot below. Phase 1 now adds in-memory customers, orders, support cases, and local APIs; see [the current domain guide](../docs/phase-1-domain.md) for the implemented behavior. The agent and AWS target remains future work.

## What we are building

NovaMind is a planned customer-resolution platform. We are building it to learn the concepts, explain the decisions in interviews, and eventually prove that it works in a real AWS account. Today it is only a local React page and FastAPI health endpoint.

## The business problem

A customer says: "I received the wrong laptop. Here is my invoice and a photo. I want a replacement."

A useful system must connect that request to the correct order, inspect evidence, check policy and stock, propose an allowed resolution, obtain approval, perform the action, and verify the result. If it cannot safely finish, it should escalate the case.

## Why this is more than a chatbot

A chatbot can explain a policy or suggest what to do. Our target system must also maintain a case and complete a verified business outcome. A convincing answer is not proof that a replacement was created. These resolution capabilities are planned; none is implemented today.

## What Agentic AI will mean here

The future Resolution Agent will interpret the customer's goal, decide which information it needs, call approved tools, observe the results, and decide the next step within application limits. It must stop or escalate when evidence or permission is missing. The model does not own the authorization boundary.

## Why one Resolution Agent

One agent gives us one decision path to understand, trace, test, and explain. We will start with read-only tools before allowing actions. More agents would add coordination and failure modes; we will consider them only for a reviewed need.

## What a tool is

A tool is a narrow function the agent may request, such as `get_order()`, `get_policy()`, or `check_inventory()`. The application must validate the inputs, permissions, and result. Those names are plans, not functions that exist in this repository.

## Why business rules are deterministic

The same verified facts and policy should produce the same eligibility decision. Code should enforce rules such as ownership, time limits, and approval requirements. The model can help interpret a request, but its wording should not change who is entitled to a refund or replacement.

## Why the LLM cannot authorize refunds or replacements

An LLM produces suggestions and may be wrong or influenced by customer content. An invoice, image, or message is evidence, not permission. The application must check authenticated identity, policy, and required human approval against the exact proposed action.

## PROPOSE → AUTHORIZE → EXECUTE → VERIFY

1. **PROPOSE:** describe the action and its supporting evidence.
2. **AUTHORIZE:** application rules check permission and obtain human approval when required.
3. **EXECUTE:** an approved tool performs the authorized action, with protection against duplicate execution on retry.
4. **VERIFY:** inspect the actual action status before declaring success. A timeout may mean the outcome is unknown, not that the action failed.

This is a future design rule. The words on today's welcome page do not implement it.

## Current local architecture

```text
Browser -> React/Vite -> local /api proxy -> FastAPI -> /api/health
```

The browser shows a welcome page. Its button asks the API whether the process responds. The API returns a small JSON response. This proves local connectivity only. There is no customer domain, agent, Bedrock connection, database, or AWS deployment.

## Target architecture

```text
React -> FastAPI -> Application services -> Resolution Agent
      -> Strands -> Amazon Bedrock -> approved business tools
      -> domain rules -> repositories/providers -> AWS persistence
```

This is a conceptual future flow. Application services coordinate a use case. Domain rules decide what the business permits. Repositories/providers handle storage and external systems. Strands is the preferred agent framework; Bedrock is the intended model provider. See the [target design](../docs/architecture/target-v2.md).

## What is implemented and what is planned

Implemented: the welcome UI, local proxy, health endpoint, dependency definitions, runtime selections, and local verification evidence. The scaffold was built before explicit approval, then audited and explicitly accepted as Phase 0B. See [project evolution](../PROJECT-EVOLUTION.md).

Planned: customer cases, tools, the Resolution Agent, document/image analysis, approvals, actions, evaluations, product APIs/UI, identity, AWS persistence and deployment, observability, and CI/CD. Local checks do not establish security, scalability, or production readiness.

## Phase 0 concepts

- **Frontend versus backend:** the browser presents information; the API is the server boundary for application operations.
- **Application versus domain:** application services coordinate a use case; domain rules decide what the business permits.
- **Agent versus tool:** the agent chooses a next step; a tool exposes a bounded operation. The model cannot grant itself permissions.
- **Proposal versus authorization:** a suggestion is not permission to spend money or change an order.
- **Execution versus verification:** requesting an action does not prove that it completed.
- **Local check versus deployment check:** a process responding on your computer does not prove that identity, networking, permissions, and storage work in AWS.

## Increment record template

For each future increment, record the concept, why it belongs in the architecture, how the code implements it, how it was tested, and one failure you can diagnose.

- Concept and problem:
- Implemented behavior:
- Decision and alternatives:
- Verification command and observed result:
- Failure scenario and diagnosis:
- Remaining limitations:

## 60-Second Project Explanation

NovaMind is a customer-resolution platform I am building to learn agentic AI and deploy a real application on AWS. A customer might report a wrong laptop and attach an invoice and photo. The target is one Resolution Agent that gathers evidence through approved tools, checks deterministic business rules, and proposes a resolution. The application, with human approval where required, authorizes sensitive actions. It then executes and verifies the result before resolving the case. Today I have only a React/Vite welcome page connected to a FastAPI health endpoint. That scaffold was built before explicit approval, then audited and accepted as the local baseline. The agent, business workflow, and AWS deployment are still planned, not production verified.

## 10 Questions I Should Be Able to Answer

1. What customer problem does NovaMind aim to solve?
2. How is a verified resolution different from a chatbot answer?
3. What works today, and what is only planned?
4. Why begin with one Resolution Agent and read-only tools?
5. What is a tool, and who validates its inputs and permissions?
6. Which decisions belong in deterministic business rules?
7. Why can neither the LLM nor an uploaded document authorize a refund?
8. What happens in PROPOSE, AUTHORIZE, EXECUTE, and VERIFY, and which layer owns each step?
9. What should happen if an action times out after a downstream system accepted it?
10. What evidence would prove a successful AWS deployment, beyond a local health check?
