import { auditTools, auditErrors } from './auditContracts.js';

export const operationLabels = Object.freeze({ http: 'HTTP requests', 'agent.loop': 'Agent loop',
  'agent.graph': 'Agent graph', tool: 'Tools', llm: 'Model requests', 'hitl.start': 'Workflow starts',
  'hitl.resume': 'Workflow resumes', 'proposal.create': 'Proposal creation',
  'proposal.review': 'Human reviews', eligibility: 'Eligibility assessments' });
const operation = (value) => typeof value === 'string' && Object.hasOwn(operationLabels, value);
const count = (value) => Number.isSafeInteger(value) && value >= 0;
const duration = (value) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= Number.MAX_SAFE_INTEGER;
const uuid = (value) => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
const timestamp = (value) => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T/.test(value) && Number.isFinite(Date.parse(value));
const oneOf = (values) => (value) => values.includes(value);
const optionalFields = {
  workflow_id: uuid, case_id: uuid, proposal_id: uuid, reviewer_user_id: uuid,
  tool: oneOf([...auditTools, 'unknown']), status: oneOf(['INFORMATIONAL', 'HUMAN_REVIEW_REQUIRED', 'FAILED',
    'REVIEW_REQUIRED', 'REVIEWED', 'COMPLETED', 'PENDING_REVIEW', 'APPROVED', 'REJECTED']),
  provider: oneOf(['gemini', 'fake', 'other']),
  invalid_response_stage: oneOf(['provider_envelope', 'json_guardrail', 'output_schema']),
  method: oneOf(['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS']),
  status_code: (value) => count(value) && value >= 100 && value <= 599,
  provider_status_code: (value) => count(value) && value >= 100 && value <= 599,
  input_tokens: (value) => count(value) && value <= 1e9,
  output_tokens: (value) => count(value) && value <= 1e9,
  total_tokens: (value) => count(value) && value <= 1e9,
  output_token_limit: (value) => count(value) && value <= 1e9,
  provider_called: (value) => typeof value === 'boolean',
  reviewer_authenticated: (value) => typeof value === 'boolean',
};
const metricCounts = ['count', 'errors', 'provider_calls', 'input_tokens_known', 'output_tokens_known',
  'total_tokens_known', 'input_tokens_unknown_calls', 'output_tokens_unknown_calls', 'total_tokens_unknown_calls'];
function check(valid) {
  if (!valid) {
    const error = new Error('Diagnostics metadata could not be verified.');
    error.code = 'INVALID_DIAGNOSTICS_RESPONSE';
    throw error;
  }
}

export function diagnosticsSnapshot(payload) {
  check(payload?.scope === 'process_metadata_only' && timestamp(payload.snapshot_at)
    && Array.isArray(payload.events) && payload.events.length <= 50
    && payload.metrics && typeof payload.metrics === 'object' && !Array.isArray(payload.metrics)
    && count(payload.evicted_events) && count(payload.omitted_events) && count(payload.logging_failures));
  const metrics = Object.entries(payload.metrics).map(([name, metric]) => {
    check(operation(name) && metric && metricCounts.every((key) => count(metric[key]))
      && duration(metric.duration_ms_total) && duration(metric.duration_ms_max)
      && metric.errors <= metric.count && metric.provider_calls <= metric.count
      && metric.duration_ms_max <= metric.duration_ms_total
      && ['input', 'output', 'total'].every((part) => metric[`${part}_tokens_unknown_calls`] <= metric.provider_calls));
    return { operation: name, ...Object.fromEntries(metricCounts.map((key) => [key, metric[key]])),
      duration_ms_total: metric.duration_ms_total, duration_ms_max: metric.duration_ms_max };
  });
  const events = payload.events.map((event) => {
    check(event && operation(event.operation) && timestamp(event.timestamp) && uuid(event.trace_id)
      && uuid(event.span_id) && (event.parent_span_id === null || uuid(event.parent_span_id))
      && duration(event.duration_ms) && (event.error === null || [...auditErrors, 'ERROR'].includes(event.error)));
    const projected = { operation: event.operation, timestamp: event.timestamp, trace_id: event.trace_id,
      span_id: event.span_id, parent_span_id: event.parent_span_id, duration_ms: event.duration_ms, error: event.error };
    for (const [key, validate] of Object.entries(optionalFields)) {
      const value = event[key] ?? null;
      check(value === null || validate(value));
      projected[key] = value;
    }
    // No schema error objects, model text, headers, credentials or generic payloads.
    return projected;
  });
  return { snapshot_at: payload.snapshot_at, metrics, events, evicted_events: payload.evicted_events,
    omitted_events: payload.omitted_events, logging_failures: payload.logging_failures };
}
