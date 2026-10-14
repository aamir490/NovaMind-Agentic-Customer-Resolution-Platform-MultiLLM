# Project evolution

## From V1 to a separate V2

```text
V1: Chatbot_text_image
Streamlit + Bedrock reference prototype
    -> Architecture assessment
    -> Decision to create a separate V2 repository
    -> Phase 0A: Architecture/planning
    -> Phase 0B: Minimal React + FastAPI scaffold
    -> Phase 0C: Repository and documentation normalization
```

V1 is a reference prototype. V2 is NovaMind Agentic Customer Resolution Platform, built for learning, interview explanation, and eventual real AWS deployment verification. The separate repository avoids carrying the whole prototype into a different architecture. No V1 source has been copied; individual ideas or components may be considered only after review. See [ADR-001](docs/decisions/ADR-001-separate-v2-repository.md).

The labels Phase 0A and Phase 0B distinguish planning from implementation retrospectively. The React/Vite welcome page, FastAPI health endpoint, local proxy, dependencies, and smoke checks were created during Phase 0 **before explicit approval of application implementation**. That was scope overreach. The assistant also wrote scope text that included the scaffold; that text was not user authorization.

A strict read-only audit was performed. The user reviewed it with ChatGPT and explicitly chose to retain the small, isolated scaffold as the Phase 0B local baseline instead of deleting and recreating it. This later acceptance does not erase the original scope overreach. See [ADR-002](docs/decisions/ADR-002-react-fastapi-local-foundation.md).

Phase 0C authorizes structure, documentation, learning material, and local verification with the already installed environment. It does not authorize an agent, business logic, AWS integration, new packages, or an initial commit. The scaffold is only a local development foundation; these technologies are not production verified for this project.

## Phase 1 — Application domain foundation

The user separately authorized Phase 1: minimum customer, order, support-case, status, validation, and local API behavior. The implementation uses in-memory records and focused domain/HTTP tests, preserving the health scaffold. No AI, AWS, authentication, infrastructure, or dependency additions are part of this increment. See [Phase 1](docs/phase-1-domain.md).

## Milestone roadmap — domain/API foundation started; later capabilities planned

Phase 3 was explicitly authorized for a local read-only case-review UI. It now displays existing context, inventory/policy data, assessment assumptions, and backend eligibility/denial results. No business-rule logic moved to the browser. See [Phase 3](docs/phase-3-case-review.md). Phase 4 remains unstarted.

Phase 2 was subsequently authorized for deterministic business operations only. Local inventory/policy fixtures and read-only eligibility APIs now exist; no agent, authorization, or execution was added. See [Phase 2](docs/phase-2-business-operations.md). Later milestones remain subject to separate approval.

```text
Business domain
    -> tools
    -> Resolution Agent
    -> multimodal evidence
    -> human approval
    -> actions
    -> evaluations
    -> FastAPI APIs
    -> React product UI
    -> AWS persistence
    -> container deployment
    -> observability
    -> CI/CD
    -> production verification
```

Phase 1 begins the business-domain foundation and its minimum local APIs. Later FastAPI APIs and React product UI mean further customer-resolution capabilities beyond these records. Each milestone needs its own scope, learning explanation, implementation review, and verification evidence. This roadmap does not authorize beginning Phase 2.
