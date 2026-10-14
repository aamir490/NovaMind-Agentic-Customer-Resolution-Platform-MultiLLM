# Phase 17 — Containerization

Container configuration and focused local checks are implemented. **14/14 Phase 17 tests passed**, both Compose configurations validated, and the frontend built offline using installed dependencies. **Image builds and running-container verification remain unverified:** Docker CLI/Compose are installed, but the Docker Desktop Linux engine is unavailable (`dockerDesktopLinuxEngine` pipe missing). No image pull, package install, registry call, container deployment, or AWS operation ran. Phase 18 is not started.

## Packaging and routing

- `backend/Dockerfile` uses Python 3.12 slim Bookworm stages. The dependency stage installs the existing exact-version `requirements.txt` into a virtual environment using binary wheels only. The runtime copies the environment, application, knowledge fixtures, and small container bootstrap/probe modules. No compiler, wheel cache, host virtual environment, tests, or evaluation datasets are copied.
- `frontend/Dockerfile` uses Node.js 24 and the existing npm lockfile to build static Vite assets. Its final Nginx Alpine stage contains the build output and routing configuration; Node, npm, source code, and node_modules are absent from that stage. Nginx serves the built application rather than running a Vite development/preview server.
- Both runtime processes use UID/GID **10001:10001**, high internal ports, and exec-form commands. Compose uses read-only root filesystems, dropped capabilities, no-new-privileges, init/reaping, and bounded writable `/tmp` tmpfs mounts. No host source/data directory or Docker socket is mounted.
- The backend runs exactly **one Uvicorn worker**, without reload or trusted proxy headers, preserving the in-memory domain/HITL boundary. Uvicorn access logs and Nginx access logs are disabled; existing Phase 16 application telemetry remains available with its original behavior. Nginx operational errors still go to stderr and are not application telemetry.
- Only the frontend publishes a host port: `127.0.0.1:8080` by default. In the base configuration, the backend belongs only to the internal `application` network, with no published port or external egress. The frontend joins both `application` (for `/api/` proxying to the backend) and the non-internal bridge network `edge` (for published-port connectivity). The frontend therefore has an external network path, while host publication remains loopback-only. Browser requests remain same-origin; no new CORS configuration is needed. The optional Gemini overlay described below explicitly enables backend egress.
- Nginx forwards `/api/` with the original path/query and bearer header to `backend:8000`, preserves API status codes and trace headers, and disables upstream retries so writes/reviews are not replayed. Docker's local DNS resolver refreshes the backend address after recreation. Missing `/assets/` files return 404; other frontend routes use the SPA fallback. Fingerprinted assets are cacheable; HTML is revalidated. Backend `/docs` and `/openapi.json` are not exposed through this frontend proxy, consistent with the earlier Vite `/api` proxy.

Each image has its own narrow build context (`backend/` or `frontend/`) and allowlist `.dockerignore`. Root files—including the three pre-existing untracked JSON files—are outside both contexts and were not read or changed. Environment files, private keys, conversation stores, caches, local dependency directories, and host builds are excluded. No build argument or frontend variable contains credentials.

## Environment and authentication

The optional `.env.container.example` contains only `NOVAMIND_PORT=8080` and configuration comments. Pass it with `--env-file`, or copy it to a local ignored `.env` and change the port. Runtime images set tracing exports off; base Compose does not pass AWS/Gemini credentials or inherit arbitrary host environment variables.

The backend container factory delegates to the existing `create_app()`. With no `NOVAMIND_AUTH_FILE`, authentication retains the original empty registry: public `/api/health` works, while business requests return 401. The existing frontend still has no login/bearer UI; containerization does not introduce an anonymous fallback.

For authenticated local API clients, the optional `compose.auth.yaml` mounts a host-provided credential file as a read-only Compose secret at `/run/secrets/novamind_auth`. `NOVAMIND_AUTH_SOURCE` is the host file path, not the credential value. The factory validates entries using existing Phase 13 `LocalCredential` and `LocalAuthenticationProvider`; roles, customer bindings, duplicate-token rejection, ownership, and authenticated review are unchanged.

The file is a JSON array of existing credential objects, for example this **shape only** (replace the token with a host-generated unpredictable value):

```json
[
  {
    "token": "<host-generated-unpredictable-local-token>",
    "identity": {
      "user_id": "00000000-0000-4000-8000-000000000017",
      "role": "ADMIN"
    }
  }
]
```

Keep the real file outside the repository, or in the ignored `.container-secrets/` directory. Provision it deliberately; no default credential file or token is generated by this phase. On Linux, arrange host ownership/permissions so runtime UID 10001 can read the mounted file; Compose file secrets do not reliably remap host UID/mode. Docker Desktop file-sharing permissions require runtime verification. These local mounted files are not an encrypted secret manager.

Configured paths must be absolute. Missing/unreadable files, malformed or duplicate-key JSON, invalid identity/token shapes, duplicate/conflicting credentials, files larger than 65,536 bytes, and lists longer than 1,000 entries stop startup with a fixed error without printing token values or paths. No invalid-configuration fallback grants access. Empty lists are allowed and remain default-deny. File loading occurs only at startup; recreate the backend to load changes. No token expiry/rotation, customer seeding, or durable business-state provisioning is added.

## Optional Gemini provider wiring (Phase 17A.4a)

Container startup can now inject the existing `GeminiProvider` as `create_app(local_provider_factory=...)`. Set **both** `GEMINI_API_KEY` and `GEMINI_MODEL` in the backend process environment through trusted runtime configuration. Unlike direct adapter use, container startup requires an explicit model rather than selecting the adapter's default. The model must be a Gemini model ID (optionally prefixed with `models/`). No key is stored in source, a build argument, the image, or frontend configuration; the bootstrap does not load `.env` files.

When both variables are absent, runs remain unavailable. Partial, blank, oversized, or malformed configuration and SDK construction failures stop startup with a fixed error, without echoing secrets. Authentication is validated first and continues to use the existing mounted credential file; configuring Gemini alone grants no API access. Startup constructs one shared provider client without calling Gemini, supplies it through the existing factory, and closes it after the existing application lifespan drains run workers. Key validity, model availability, quotas, and connectivity can only be determined by a provider request; these are not startup health guarantees. Provider failures retain the existing sanitized workflow failure handling.

For Compose, use the opt-in `compose.gemini.yaml` alongside the base and auth files. Inject the key into the invoking process with your trusted secret mechanism, set `GEMINI_MODEL` to your available model ID, and set `NOVAMIND_AUTH_SOURCE` as documented above. The overlay requires both Gemini variables and passes them only to the backend at runtime. It also attaches the backend to the existing `edge` bridge so Gemini requests can leave the container; this grants general outbound connectivity, not a Gemini-specific network allowlist. The base stack remains isolated and no backend host port is added.

```powershell
docker compose --env-file .env.container.example -f compose.yaml -f compose.auth.yaml -f compose.gemini.yaml config --quiet
docker compose --env-file .env.container.example -f compose.yaml -f compose.auth.yaml -f compose.gemini.yaml up --build -d --wait
```

These commands are operator instructions, not verification performed for this change. Keep actual credentials out of shell command history, repository files, and logs. Environment injection is visible to trusted container/host administrators; do not export rendered Compose configuration or container inspection output containing it. Recreate the backend to rotate configuration (its existing in-memory state is lost). No secret-management service or new dependency is added.

The existing HTTP provider label remains `local_scripted` for any configured local factory; the actual configured adapter is Gemini. This wiring deliberately leaves API schemas and frontend labels unchanged. No agent, graph, business rule, approval boundary, or automatic action is changed. Focused tests substitute the SDK transport and exercise the real adapter and authenticated run route; no live Gemini or Docker verification is claimed.

## Optional Groq provider wiring (Phase 17B.5a)

Groq is available through the same local provider factory with explicit
`NOVAMIND_LLM_PROVIDER=groq`, runtime-injected `GROQ_API_KEY`, and optional
`GROQ_MODEL` (default `openai/gpt-oss-20b`). Use `compose.groq.yaml` instead of the
Gemini overlay. Without a selector, the existing Gemini configuration behavior
above remains unchanged. See [Groq configuration, JSON compatibility, tests, and
limitations](phase-17b5a-groq.md). Live Groq and Docker execution are not verified
by the mocked tests; Bedrock is not implemented.

## Health and local verification

The backend Docker health check calls only `http://127.0.0.1:8000/api/health`, with a two-second timeout, bounded response, disabled environment proxies, and no redirects. It validates the existing health response. It does not check a model, database, external service, authorization readiness, or workflow recovery.

The frontend Docker health check uses its local `/healthz`. Compose waits for initial backend health before starting the frontend; this does not promise automatic recovery when a dependency later becomes unhealthy. The separate loopback smoke checker verifies the page, built JavaScript, frontend health, proxied backend health, anonymous API denial, trace header, and missing-asset 404 behavior. It performs read-only requests, follows no redirects, uses no credentials/proxies, and never contacts a model/cloud service.

After Docker Desktop's Linux engine is available, the intended commands from the repository root are:

```powershell
docker compose --env-file .env.container.example config --quiet
docker compose --env-file .env.container.example build
docker compose --env-file .env.container.example up --no-build --pull never -d --wait
docker compose --env-file .env.container.example exec -T frontend nginx -t
./backend/.venv/Scripts/python.exe -B scripts/container_smoke.py --port 8080
docker compose --env-file .env.container.example ps
```

The **build command requires preloaded base images and cached packages or permitted registry access**. It was not run in this offline session. BuildKit dependency download access is separate from the runtime internal network. Runtime `up --no-build --pull never` deliberately cannot silently fetch or rebuild missing images. If using a different port, pass the same port to the checker. Stop the local stack with `docker compose --env-file .env.container.example down`; stopping/recreating the backend loses its in-memory records/checkpoints as before.

For the optional credential overlay, set the host path and use both files consistently for startup/management:

```powershell
$env:NOVAMIND_AUTH_SOURCE = 'C:/private/novamind-auth.json'
docker compose --env-file .env.container.example -f compose.yaml -f compose.auth.yaml config --quiet
docker compose --env-file .env.container.example -f compose.yaml -f compose.auth.yaml up --no-build --pull never -d --wait
```

No real secret file was used during verification. The overlay was syntax-validated with an existing non-secret file as a temporary path placeholder; containers were not launched with it.

## Recorded focused checks

- `./backend/.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_containers.py -v` — **14/14 passed**. Covers default-deny/health/trace startup, mounted credentials and roles, isolated state after restart, safe configuration failure, bounded loopback probes, image/Compose/context boundaries, API routing/retry configuration, and the smoke checker's positive/negative outcomes. Two initial probe-test mock setup errors were corrected before the passing run.
- `docker compose --env-file .env.container.example -f compose.yaml config --quiet` — **passed**.
- The same config check with `-f compose.auth.yaml` and an explicit non-secret source path — **passed**.
- `npm.cmd --offline run build` in `frontend/` — **passed**, Vite 8.3.1, 17 modules. No install or registry query ran; output is ignored `frontend/dist/`.
- `git diff --check` — **passed**, line-ending normalization warnings only.
- Docker image builds, Nginx binary/config validation, live container health/startup, runtime permissions, Linux wheel installation, and the end-to-end smoke checker against running containers — **not run**, engine unavailable. Static configuration checks and mocked health checks do not establish those results.

Only Phase 17 focused tests/checks ran. No earlier test suite, Phase 15 evaluation, full regression, AWS/Gemini call, image push, or deployment ran.

## Dependencies, retained limits, and files

No Python/npm dependency or lockfile changes. Tests reuse installed HTTPX/PyYAML; bootstrap and probes use the standard library plus existing application contracts. Python/Node/Nginx base images are required packaging inputs, not additional application packages. Their tags are version-family tags, not immutable digests; exact image reproducibility, availability, vulnerabilities, sizes, and architecture compatibility remain unverified. No image-size or production-readiness claim is made. Linux builds intentionally fail if a pinned Python package lacks an appropriate binary wheel; no compiler or dependency upgrade is silently added.

Application modules under `backend/app`, frontend source, business rules, tool allowlists, guardrails, authentication/ownership, human review, evaluation, and observability code are unchanged. Container bootstrap is isolated under `backend/container`. Existing agent/HITL Python entry points remain available; no new agent HTTP endpoints or autonomous execution are added. Approval still executes nothing. Business data, HITL checkpoints, audit history, and telemetry remain in-memory; SQLite conversation storage is still an explicitly provisioned programmatic feature, not automatically mounted or initialized by the server. Containerization adds no persistence, replication, TLS, production identity, or scaling guarantees.

Added backend/frontend Dockerfiles and context allowlists, Nginx configuration, base/optional-auth Compose files, an environment example, container bootstrap/health modules, `scripts/container_smoke.py`, and `tests/test_containers.py`. Updated relevant READMEs, local-development guidance, both roadmaps, the Phase 16 status pointer, and this guide. The three untracked root JSON files remain untouched. **No AWS deployment or Phase 18 work was started.**
