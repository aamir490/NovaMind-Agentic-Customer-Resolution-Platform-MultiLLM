import { auditBinding, auditCases, auditRuns, auditPage } from './auditContracts.js';
import { diagnosticsSnapshot } from './diagnosticsContracts.js';

// Same-origin authenticated reads, explicit run starts, and explicit human decisions.
const isText = (value) => typeof value === 'string' && value.trim().length > 0;
const isCount = (value) => Number.isSafeInteger(value) && value >= 0;
const isTextList = (value) => Array.isArray(value) && value.every(isText);
const isUuid = (value) => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
const isOptionalUuid = (value) => value === null || isUuid(value);
const isOptionalText = (value) => value === null || typeof value === 'string';
const isTimestamp = (value) => typeof value === 'string' && Number.isFinite(Date.parse(value));
const runStatuses = ['RUNNING', 'RESUMING', 'REVIEW_REQUIRED', 'REVIEWED', 'COMPLETED', 'FAILED'];
const eventStates = [...runStatuses, 'STARTED', 'PAUSED'];
const eventNodes = ['load_case', 'reason', 'execute_tool', 'record_result', 'finalize',
  'fail_safely', 'prepare_review', 'human_review'];

function requireResponse(valid, resource) {
  if (!valid) throw new Error(`The API returned an invalid ${resource} response. No result was loaded`);
}

function myCase(supportCase) {
  requireResponse(isUuid(supportCase?.id) && isUuid(supportCase.order_id)
    && isText(supportCase.subject) && isText(supportCase.description)
    && isText(supportCase.status) && isTimestamp(supportCase.created_at)
    && isTimestamp(supportCase.updated_at), 'my support request');
  // Only customer-facing fields; never expose workflow or reviewer metadata.
  return { id: supportCase.id, order_id: supportCase.order_id,
    subject: supportCase.subject, description: supportCase.description,
    status: supportCase.status, created_at: supportCase.created_at,
    updated_at: supportCase.updated_at };
}

function knowledgeEvidence(event) {
  // Optional evidence must never interrupt run tracking. Missing or invalid
  // evidence is unavailable, distinct from a confirmed retrieval with no hits.
  if (event.kind !== 'TOOL' || event.tool !== 'search_knowledge' || event.ok !== true
      || event.state !== 'COMPLETED' || event.retrieval_method !== 'local_token_cosine_v1'
      || !Array.isArray(event.evidence) || event.evidence.length > 5) return null;
  const boundedText = (value, max) => isText(value) && Array.from(value).length <= max;
  const references = new Set();
  const evidence = [];
  for (const hit of event.evidence) {
    const chunk = hit?.chunk;
    const source = chunk?.source;
    if (!chunk || !source || chunk.trust !== 'untrusted_information' || source.authority !== 'reference_only'
        || typeof chunk.chunk_id !== 'string' || !/^[a-f0-9]{64}$/.test(chunk.chunk_id)
        || references.has(chunk.chunk_id) || !boundedText(chunk.text, 600)
        || !isCount(chunk.start_char) || !isCount(chunk.end_char)
        || chunk.end_char - chunk.start_char !== Array.from(chunk.text).length
        || typeof source.document_id !== 'string' || !/^[a-z0-9-]{1,80}$/.test(source.document_id)
        || !boundedText(source.title, 160) || !boundedText(source.version, 160)
        || typeof source.source_uri !== 'string' || !/^local-knowledge:\/\/[a-z0-9-]{1,80}$/.test(source.source_uri)) return null;
    references.add(chunk.chunk_id);
    // Closed display projection: no query, score, prompts, or arbitrary payloads.
    evidence.push({ reference: chunk.chunk_id, documentId: source.document_id,
      title: source.title, version: source.version, source: source.source_uri, snippet: chunk.text });
  }
  return evidence;
}

function runSnapshot(result, { instance_id, case_id, run_id }) {
  requireResponse(result?.instance_id === instance_id && result.case_id === case_id && isUuid(result.run_id)
    && (run_id === undefined || result.run_id === run_id) && runStatuses.includes(result.status)
    && result.provider === 'local_scripted' && result.actions_executed === false
    && isOptionalUuid(result.workflow_id) && isOptionalUuid(result.conversation_id) && isOptionalUuid(result.trace_id)
    && isTimestamp(result.created_at) && isTimestamp(result.updated_at)
    && isCount(result.last_sequence) && isOptionalText(result.message) && isOptionalText(result.error), 'run snapshot');
  requireResponse(result.reviewed_status === null || ['APPROVED', 'REJECTED'].includes(result.reviewed_status), 'review outcome');
  const review = result.review;
  requireResponse(review === null || (review && isUuid(review.workflow_id) && review.workflow_id === result.workflow_id
    && review.case_id === case_id && isUuid(review.proposal_id) && isUuid(review.review_id)
    && ['return', 'refund', 'replacement'].includes(review.action) && review.status === 'PENDING_REVIEW'
    && isText(review.rationale) && review.rationale.length <= 4000), 'run review');
  // Explicit display projection; no raw workflow state or unrestricted response dump.
  return { instance_id: result.instance_id, run_id: result.run_id, case_id: result.case_id,
    workflow_id: result.workflow_id, conversation_id: result.conversation_id, trace_id: result.trace_id,
    provider: result.provider, status: result.status, created_at: result.created_at, updated_at: result.updated_at,
    last_sequence: result.last_sequence, message: result.message, error: result.error,
    review: review === null ? null : { workflow_id: review.workflow_id, case_id: review.case_id,
      proposal_id: review.proposal_id, review_id: review.review_id, action: review.action,
      rationale: review.rationale, status: review.status },
    reviewed_status: result.reviewed_status,
    actions_executed: result.actions_executed };
}

export function createApi(fetcher = globalThis.fetch, { token = '', onUnauthorized } = {}) {
  // One in-memory client per identity. Never persist credentials or put them in URLs.
  const lifetime = new AbortController();

  async function request(path, { method = 'GET', body, signal } = {}) {
    if (lifetime.signal.aborted) throw new Error('This browser session has ended.');
    const response = await fetcher(`/api${path}`, {
      method, cache: 'no-store', credentials: 'omit', redirect: 'error',
      headers: { ...(globalThis.location?.hostname?.endsWith('.ngrok-free.app') ? { 'ngrok-skip-browser-warning': 'true' } : {}), ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(body ? { 'Content-Type': 'application/json' } : {}) },
      ...(body ? { body: JSON.stringify(body) } : {}),
      signal: AbortSignal.any([AbortSignal.timeout(5000), lifetime.signal, ...(signal ? [signal] : [])]),
    });
    if (!response.ok) {
      let detail;
      try { detail = (await response.json()).detail; } catch { /* Use status fallback. */ }
      const message = Array.isArray(detail) ? detail.map((item) => item.msg).join('; ') : detail;
      const error = new Error(typeof message === 'string' ? message : `API request failed (${response.status})`);
      error.status = response.status;
      error.code = typeof detail === 'string' ? detail : null;
      if (response.status === 401) onUnauthorized?.();
      throw error;
    }
    return response.json();
  }
  const get = (path, options) => request(path, options);
  return {
    dispose() {
      token = '';
      lifetime.abort();
    },
    async identity() {
      const payload = await get('/identity');
      const identity = payload?.identity;
      if (!identity || !isUuid(identity.user_id)
          || !['CUSTOMER', 'REVIEWER', 'ADMIN'].includes(identity.role)
          || (identity.role === 'CUSTOMER'
            ? !isUuid(identity.customer_id)
            : identity.customer_id !== null)
          || !isUuid(payload.instance_id) || !['unavailable', 'local_scripted'].includes(payload.provider)
          || typeof payload.memory_available !== 'boolean') {
        throw new Error('Unexpected identity response.');
      }
      // Display only identity fields confirmed by the backend, never token contents.
      return { user_id: identity.user_id, role: identity.role, customer_id: identity.customer_id,
        instance_id: payload.instance_id, provider: payload.provider, memory_available: payload.memory_available };
    },
    async pendingReviews(instanceId, options) {
      const page = await get('/reviews', options);
      requireResponse(page?.instance_id === instanceId && isTimestamp(page.snapshot_at)
        && page.scope === 'pending_reviews_in_this_process'
        && Array.isArray(page.items) && page.items.length <= 1024, 'pending review queue');
      const runIds = new Set();
      const items = page.items.map((item) => {
        const review = item?.review;
        requireResponse(item && isUuid(item.run_id) && !runIds.has(item.run_id)
          && isUuid(item.case_id) && isUuid(item.workflow_id) && isText(item.case_subject)
          && item.status === 'REVIEW_REQUIRED' && item.actions_executed === false && isTimestamp(item.updated_at)
          && review && review.case_id === item.case_id && review.workflow_id === item.workflow_id
          && isUuid(review.proposal_id) && isUuid(review.review_id) && review.status === 'PENDING_REVIEW'
          && ['return', 'refund', 'replacement'].includes(review.action)
          && isText(review.rationale) && Array.from(review.rationale).length <= 4000, 'pending review');
        runIds.add(item.run_id);
        return { runId: item.run_id, caseId: item.case_id, caseSubject: item.case_subject,
          workflowId: item.workflow_id, updatedAt: item.updated_at, status: item.status,
          proposalId: review.proposal_id, reviewId: review.review_id, action: review.action,
          rationale: review.rationale, reviewStatus: review.status };
      });
      return { items, snapshotAt: page.snapshot_at };
    },
    async decideReview(instanceId, item, decision, options) {
      requireResponse(['APPROVE', 'REJECT'].includes(decision), 'human decision');
      // One explicit POST. A lost response must never trigger an automatic replay.
      const result = await request(`/runs/${encodeURIComponent(item.runId)}/decision`, {
        method: 'POST', signal: options?.signal,
        body: { instance_id: instanceId, case_id: item.caseId, workflow_id: item.workflowId,
          proposal_id: item.proposalId, review_id: item.reviewId, decision },
      });
      const snapshot = runSnapshot(result, { instance_id: instanceId, case_id: item.caseId, run_id: item.runId });
      requireResponse(snapshot.workflow_id === item.workflowId && snapshot.review?.review_id === item.reviewId
        && snapshot.review.proposal_id === item.proposalId, 'decision binding');
      return snapshot;
    },
    async startRun({ instance_id, request_id, case_id, message }) {
      // Exactly one POST per explicit submission; never retry a potentially committed run.
      const result = await request('/runs', { method: 'POST', body: { instance_id, request_id, case_id, message } });
      return runSnapshot(result, { instance_id, case_id });
    },
    async runSnapshot(binding, options) {
      return runSnapshot(await get(`/runs/${encodeURIComponent(binding.run_id)}`, options), binding);
    },
    async runEvents(binding, after, options) {
      const page = await get(`/runs/${encodeURIComponent(binding.run_id)}/events?after=${after}&limit=100`, options);
      const snapshot = runSnapshot(page?.snapshot, binding);
      requireResponse(Array.isArray(page.events) && page.events.length <= 100
        && isCount(page.next_after) && page.next_after >= after && page.next_after <= snapshot.last_sequence
        && isCount(page.first_available_sequence) && page.first_available_sequence >= 1
        && page.first_available_sequence <= snapshot.last_sequence + 1
        && page.dropped_events === page.first_available_sequence - 1
        && page.gap === (after < page.first_available_sequence - 1)
        && page.has_more === (page.next_after < snapshot.last_sequence), 'run events');
      const events = page.events.map((event, index) => {
        requireResponse(event?.run_id === binding.run_id && isCount(event.sequence)
          && event.sequence === Math.max(after + 1, page.first_available_sequence) + index
          && event.sequence <= snapshot.last_sequence && isTimestamp(event.timestamp)
          && ['STATE', 'NODE', 'TOOL'].includes(event.kind) && eventStates.includes(event.state)
          && (event.node === null || eventNodes.includes(event.node)) && isOptionalText(event.tool)
          && (event.ok === null || typeof event.ok === 'boolean') && isOptionalText(event.error)
          && isOptionalUuid(event.proposal_id), 'run event');
        return { sequence: event.sequence, timestamp: event.timestamp, kind: event.kind, state: event.state,
          node: event.node, tool: event.tool, ok: event.ok, error: event.error, proposal_id: event.proposal_id,
          evidence: knowledgeEvidence(event) };
      });
      requireResponse(page.next_after === (events.at(-1)?.sequence ?? after)
        && (!page.has_more || events.length > 0), 'run event cursor');
      return { snapshot, events, next_after: page.next_after, has_more: page.has_more, gap: page.gap };
    },
    async diagnostics(options) {
      return diagnosticsSnapshot(await get('/diagnostics?limit=50', options));
    },
    async auditCases(options) {
      return auditCases(await get('/cases', options));
    },
    async auditRuns(binding, offset = 0, options) {
      auditBinding(binding, offset);
      return auditRuns(await get(`/cases/${encodeURIComponent(binding.case_id)}/runs?offset=${offset}&limit=50`, options), binding, offset);
    },
    async runAudit(binding, after = 0, options) {
      auditBinding(binding, after, true);
      return auditPage(await get(`/runs/${encodeURIComponent(binding.run_id)}/audit?after=${after}&limit=50`, options), binding, after);
    },
    // -----------------------------------------------------------------
    // Customer portal — Phase 17B.2
    // -----------------------------------------------------------------

    async listCatalog({ availableOnly = false } = {}) {
      const params = availableOnly ? '?available_only=true' : '';
      const products = await get(`/catalog${params}`);
      requireResponse(Array.isArray(products) && products.length <= 1000, 'catalog list');
      return products.map((p) => {
        requireResponse(isUuid(p?.id) && isText(p.sku) && isText(p.name) && isText(p.description)
          && typeof p.available === 'boolean'
          && p.unit_price != null && isText(p.unit_price.currency)
          && typeof p.unit_price.amount === 'string' && p.unit_price.amount.length <= 20,
          'catalog product');
        return { id: p.id, sku: p.sku, name: p.name, description: p.description,
          available: p.available,
          unit_price: { amount: p.unit_price.amount, currency: p.unit_price.currency } };
      });
    },

    async getCatalogProduct(id) {
      requireResponse(isUuid(id), 'product id');
      const p = await get(`/catalog/${encodeURIComponent(id)}`);
      requireResponse(isUuid(p?.id) && isText(p.sku) && isText(p.name) && isText(p.description)
        && typeof p.available === 'boolean'
        && p.unit_price != null && isText(p.unit_price.currency)
        && typeof p.unit_price.amount === 'string', 'catalog product');
      return { id: p.id, sku: p.sku, name: p.name, description: p.description,
        available: p.available,
        unit_price: { amount: p.unit_price.amount, currency: p.unit_price.currency } };
    },

    async createMyOrder(items) {
      // SECURITY: customer_id is NEVER sent — the backend derives it from the
      // authenticated identity. Only items (sku + quantity) are accepted.
      requireResponse(Array.isArray(items) && items.length >= 1 && items.length <= 100,
        'order items');
      for (const item of items) {
        requireResponse(isText(item?.sku) && Number.isSafeInteger(item.quantity)
          && item.quantity >= 1 && item.quantity <= 1000, 'order item');
      }
      const body = { items: items.map(({ sku, quantity }) => ({ sku, quantity })) };
      const order = await request('/my/orders', { method: 'POST', body });
      requireResponse(isUuid(order?.id) && isUuid(order.customer_id)
        && Array.isArray(order.items) && order.items.length >= 1, 'created order');
      return { id: order.id, customer_id: order.customer_id,
        items: order.items.map((it) => ({ sku: it.sku, name: it.name, quantity: it.quantity })) };
    },

    async createMyCase({ orderId, subject, description }) {
      // SECURITY: customer_id is NEVER sent — the backend derives it from the
      // authenticated identity. Only order_id, subject, and description are sent.
      requireResponse(isUuid(orderId), 'case order_id');
      requireResponse(isText(subject) && Array.from(subject).length <= 200, 'case subject');
      requireResponse(isText(description) && Array.from(description).length <= 4000, 'case description');
      const body = { order_id: orderId, subject: subject.trim(), description: description.trim() };
      const supportCase = await request('/my/cases', { method: 'POST', body });
      requireResponse(isUuid(supportCase?.id) && isUuid(supportCase.customer_id)
        && isUuid(supportCase.order_id) && isText(supportCase.subject)
        && isText(supportCase.description) && isText(supportCase.status)
        && isTimestamp(supportCase.created_at), 'created case');
      return {
        id: supportCase.id,
        customer_id: supportCase.customer_id,
        order_id: supportCase.order_id,
        subject: supportCase.subject,
        description: supportCase.description,
        status: supportCase.status,
        created_at: supportCase.created_at,
      };
    },

    async listMyCases(options) {
      // The authenticated backend scopes this read. Never send customer_id.
      const cases = await get('/cases', { signal: options?.signal });
      requireResponse(Array.isArray(cases), 'my support requests list');
      return cases.map(myCase);
    },

    async getMyCase(caseId, options) {
      requireResponse(isUuid(caseId), 'case id');
      const supportCase = myCase(await get(`/cases/${encodeURIComponent(caseId)}`, { signal: options?.signal }));
      requireResponse(supportCase.id === caseId, 'my support request');
      return supportCase;
    },

    async listMyOrders() {
      const orders = await get('/my/orders');
      requireResponse(Array.isArray(orders) && orders.length <= 10000, 'my orders list');
      return orders.map((order) => {
        requireResponse(isUuid(order?.id) && isUuid(order.customer_id)
          && Array.isArray(order.items) && order.items.length >= 1, 'my order');
        return { id: order.id, customer_id: order.customer_id,
          items: order.items.map((it) => ({ sku: it.sku, name: it.name, quantity: it.quantity })) };
      });
    },

    listCases: () => get('/cases'),
    async context(id) {
      const supportCase = await get(`/cases/${encodeURIComponent(id)}`);
      const [customer, order] = await Promise.all([
        get(`/customers/${encodeURIComponent(supportCase.customer_id)}`),
        get(`/orders/${encodeURIComponent(supportCase.order_id)}`),
      ]);
      return { supportCase, customer, order };
    },
    async item(sku) {
      const inventory = await get(`/inventory/${encodeURIComponent(sku)}`);
      requireResponse(inventory?.sku === sku && isCount(inventory.available_quantity)
        && isText(inventory.policy_id), 'inventory');
      const policy = await get(`/policies/${encodeURIComponent(inventory.policy_id)}`);
      requireResponse(policy?.id === inventory.policy_id && isText(policy.version)
        && isCount(policy.return_window_days) && isTextList(policy.allowed_reasons)
        && typeof policy.change_of_mind_requires_unused === 'boolean'
        && typeof policy.refund_requires_return === 'boolean'
        && typeof policy.description === 'string', 'policy');
      return { inventory, policy };
    },
    async eligibility(orderId, inputs) {
      const submitted = { ...inputs };
      const result = await get(
        `/orders/${encodeURIComponent(orderId)}/eligibility?${new URLSearchParams(submitted)}`,
      );
      // Check the returned contract and request binding, never recompute eligibility.
      // Missing booleans must not render as invented denials or authorization.
      requireResponse(result?.order_id === orderId && result.customer_id === submitted.customer_id
        && result.sku === submitted.sku && result.reason === submitted.reason
        && result.condition === submitted.condition
        && isCount(result.requested_quantity) && result.requested_quantity === Number(submitted.quantity)
        && isCount(result.days_since_delivery) && result.days_since_delivery === Number(submitted.days_since_delivery)
        && isCount(result.purchased_quantity) && isText(result.policy_id) && isText(result.policy_version)
        && typeof result.return_eligible === 'boolean' && typeof result.refund_eligible === 'boolean'
        && typeof result.refund_requires_return === 'boolean' && isTextList(result.denial_reasons)
        && result.assessment_basis === 'local_scenario_unverified_inputs'
        && result.authorization_granted === false, 'eligibility');
      return result;
    },
  };
}

export const api = createApi();
