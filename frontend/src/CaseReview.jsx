import { useEffect, useRef, useState } from 'react';
import { api as defaultApi } from './api.js';

// Human-readable denial reason labels.
const denialLabels = {
  item_not_in_order: 'Item is not part of this order.',
  quantity_exceeds_purchased: 'Requested quantity exceeds the purchased quantity.',
  outside_return_window: 'Delivery age is outside the policy return window.',
  reason_not_allowed: 'Policy does not permit this return reason.',
  change_of_mind_requires_unused: 'Change-of-mind returns require unused condition.',
};

const label = (value) => value.replaceAll('_', ' ');

const loadingMessages = {
  cases:      'Loading available support cases\u2026',
  context:    'Loading case, customer, and order details\u2026',
  item:       'Loading inventory and applicable policy\u2026',
  assessment: 'Checking eligibility with the backend\u2026',
};

// Status badge CSS modifier — maps backend status strings to design-system tones.
function statusClass(status) {
  const s = (status || '').toLowerCase();
  if (s === 'open')      return 'case-status--open';
  if (s === 'resolved' || s === 'closed') return 'case-status--resolved';
  if (s === 'in_review' || s === 'in_progress') return 'case-status--active';
  return '';
}

function Detail({ title, children }) {
  return (
    <div>
      <dt>{title}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function EmptyPanel({ icon, title, children }) {
  return (
    <div className="case-empty">
      <span className="case-empty-icon" aria-hidden="true">
        {icon ?? (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round">
            <path d="M3 7V5h7l2 2h9v13H3zM8 12h8M8 16h5" />
          </svg>
        )}
      </span>
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}

// Inline spinner for loading states inside panels.
function LoadingDot() {
  return <span className="case-loading-dot" aria-hidden="true" />;
}

export default function CaseReview({ api = defaultApi }) {
  const [cases, setCases]     = useState(null);
  const [selected, setSelected] = useState('');
  const [context, setContext] = useState(null);
  const [item, setItem]       = useState(null);
  const [sku, setSku]         = useState('');
  const [inputs, setInputs]   = useState({
    quantity: '1', days_since_delivery: '', reason: '', condition: '',
  });
  const [result, setResult]   = useState(null);
  const [pending, setPending] = useState(null);
  const [failure, setFailure] = useState(null);
  const requestVersion = useRef(0);
  const busy  = pending !== null;
  const error = failure?.message || '';

  useEffect(() => () => { requestVersion.current += 1; }, [api]);

  async function run(stage, request, applyResult) {
    const version = ++requestVersion.current;
    setPending(stage);
    setFailure(null);
    try {
      const data = await request();
      if (version === requestVersion.current) applyResult(data);
    } catch (cause) {
      if (version === requestVersion.current) {
        setFailure({
          stage,
          message: `${cause.message || 'Request failed'}. Check the local backend and retry.`,
        });
      }
    } finally {
      if (version === requestVersion.current) setPending(null);
    }
  }

  function clearReview() {
    requestVersion.current += 1;
    setPending(null); setFailure(null);
    setContext(null); setItem(null); setSku(''); setResult(null);
    setInputs({ quantity: '1', days_since_delivery: '', reason: '', condition: '' });
  }

  function edit(name, value) {
    requestVersion.current += 1;
    setPending(null);
    setInputs((previous) => ({ ...previous, [name]: value }));
    setResult(null); setFailure(null);
  }

  function loadItem(nextSku) {
    setItem(null); setResult(null);
    run('item', () => api.item(nextSku), setItem);
  }

  return (
    <section className="review case-resolution" aria-labelledby="review-title" aria-busy={busy}>

      {/* ── Case selector panel ──────────────────────────────────── */}
      <div className="case-selector-panel">
        <div className="review-heading">
          <div>
            <p className="eyebrow">Operations console</p>
            <h2 id="review-title" tabIndex={-1}>Case review</h2>
          </div>
          <div className="case-toolbar-actions">
            {cases !== null && (
              <span className="case-count">
                {cases.length} {cases.length === 1 ? 'case' : 'cases'} loaded
              </span>
            )}
            <button
              className="button-secondary"
              disabled={busy}
              onClick={() => {
                clearReview(); setSelected(''); setCases(null);
                run('cases', () => api.listCases(), setCases);
              }}
            >
              {pending === 'cases' ? <><LoadingDot /> Loading\u2026</> : 'Refresh cases'}
            </button>
          </div>
        </div>

        {!!cases?.length && (
          <form
            className="case-selector-form"
            onSubmit={(event) => {
              event.preventDefault(); clearReview();
              run('context', () => api.context(selected), setContext);
            }}
          >
            <fieldset disabled={busy}>
              <label>
                Support case
                <select
                  required
                  value={selected}
                  onChange={(event) => { setSelected(event.target.value); clearReview(); }}
                >
                  <option value="">Choose a case\u2026</option>
                  {cases.map((entry) => (
                    <option key={entry.id} value={entry.id}>
                      {entry.subject} \u2014 {entry.id}
                    </option>
                  ))}
                </select>
              </label>
              <button type="submit" disabled={!selected || busy}>
                {pending === 'context' ? 'Loading\u2026' : 'Load case'}
              </button>
            </fieldset>
          </form>
        )}

        <p className="review-description">
          Read-only review workspace. No case updates, stock reservations, refunds, or approvals are performed here.
        </p>
      </div>

      {/* ── Global status / error ────────────────────────────────── */}
      {busy && pending !== 'cases' && pending !== 'context' && (
        <p className="case-notice" role="status">
          <LoadingDot />{loadingMessages[pending]}
        </p>
      )}
      {error && (
        <div className="case-error-banner" role="alert">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round"
            aria-hidden="true">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 8v4M12 16h.01" />
          </svg>
          <span>{error}</span>
        </div>
      )}

      {/* ── Pre-context states ───────────────────────────────────── */}
      {!context && !busy && !error && (
        <div className="case-panel case-start" role="status">
          {cases === null ? (
            <EmptyPanel title="Start a case review">
              Refresh cases to load the records available to your identity.
              Then choose a case to review its customer, order, and item details.
            </EmptyPanel>
          ) : cases.length === 0 ? (
            <EmptyPanel
              title="No cases available"
              icon={
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round">
                  <path d="M20 7H4a2 2 0 00-2 2v10a2 2 0 002 2h16a2 2 0 002-2V9a2 2 0 00-2-2zM16 3H8l-2 4h12z" />
                </svg>
              }
            >
              No support cases were returned. Refresh when case records are available in this process.
            </EmptyPanel>
          ) : (
            <EmptyPanel title="Select a case to begin">
              Choose a support case above and load its customer, order, and item details.
            </EmptyPanel>
          )}
        </div>
      )}

      {/* ── Loaded case content ──────────────────────────────────── */}
      {context && (
        <>
          {/* Case summary header */}
          <section className="case-panel case-summary" aria-labelledby="case-summary-title">
            <div className="case-panel-heading">
              <div>
                <p className="eyebrow">Case details</p>
                <h3 id="case-summary-title">{context.supportCase.subject}</h3>
              </div>
              <span className={`case-status ${statusClass(context.supportCase.status)}`}>
                {label(context.supportCase.status)}
              </span>
            </div>

            <p className="case-description">{context.supportCase.description}</p>

            <dl className="case-reference">
              <Detail title="Case ID">
                <span className="case-mono">{context.supportCase.id}</span>
              </Detail>
            </dl>
          </section>

          {/* Context + policy grid */}
          <div className="case-details-grid">

            {/* Customer & order context */}
            <section className="case-panel" aria-labelledby="case-context-title">
              <div className="case-panel-heading">
                <div>
                  <p className="eyebrow">Customer &amp; order</p>
                  <h3 id="case-context-title">Review context</h3>
                </div>
              </div>

              <dl className="case-facts">
                <Detail title="Customer name">{context.customer.name}</Detail>
                <Detail title="Customer ID">
                  <span className="case-mono">{context.customer.id}</span>
                </Detail>
                <Detail title="Order ID">
                  <span className="case-mono">{context.order.id}</span>
                </Detail>
              </dl>

              <div className="case-subsection">
                <h4>Ordered items</h4>
                <ul className="case-item-list">
                  {context.order.items.map((entry, index) => (
                    <li key={`${entry.sku}-${index}`}>
                      <div>
                        <strong>{entry.name}</strong>
                        <span className="case-item-sku">{entry.sku}</span>
                      </div>
                      <span className="case-item-quantity">Qty {entry.quantity}</span>
                    </li>
                  ))}
                </ul>

                <label className="case-item-select">
                  Item to review
                  <select
                    disabled={busy}
                    value={sku}
                    onChange={(event) => {
                      const next = event.target.value;
                      requestVersion.current += 1;
                      setPending(null); setFailure(null);
                      setSku(next); setItem(null); setResult(null);
                      setInputs({ quantity: '1', days_since_delivery: '', reason: '', condition: '' });
                      if (next) loadItem(next);
                    }}
                  >
                    <option value="">Choose an item\u2026</option>
                    {[...new Set(context.order.items.map((entry) => entry.sku))].map((value) => (
                      <option key={value} value={value}>{value}</option>
                    ))}
                  </select>
                </label>
              </div>
            </section>

            {/* Inventory & policy reference */}
            <section className="case-panel" aria-labelledby="case-policy-title">
              <div className="case-panel-heading">
                <div>
                  <p className="eyebrow">Policy reference</p>
                  <h3 id="case-policy-title">Inventory &amp; applicable policy</h3>
                </div>
              </div>

              {item ? (
                <>
                  {/* Stock indicator */}
                  <div className={`case-stock ${item.inventory.available_quantity > 0 ? 'case-stock--in' : 'case-stock--out'}`}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                      strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round"
                      aria-hidden="true">
                      <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
                      <polyline points="9 22 9 12 15 12 15 22" />
                    </svg>
                    <div>
                      <strong>{item.inventory.available_quantity} units</strong>
                      <span>available &mdash; {item.inventory.sku}</span>
                    </div>
                    <span className={`case-stock-badge ${item.inventory.available_quantity > 0 ? 'case-stock-badge--in' : 'case-stock-badge--out'}`}>
                      {item.inventory.available_quantity > 0 ? 'In stock' : 'Out of stock'}
                    </span>
                  </div>
                  <p className="case-caption">
                    Stock level is informational for this assessment and does not affect eligibility.
                  </p>

                  {/* Policy facts */}
                  <dl className="case-facts case-policy-facts">
                    <Detail title="Policy ID &amp; version">
                      <span className="case-mono">{item.policy.id}</span>
                      {' '}&middot; {item.policy.version}
                    </Detail>
                    <Detail title="Return window">
                      <span className="case-policy-value">{item.policy.return_window_days} days</span>
                      <span className="case-policy-note">(from delivery date, inclusive)</span>
                    </Detail>
                    <Detail title="Allowed return reasons">
                      <ul className="case-policy-reasons">
                        {item.policy.allowed_reasons.map((r) => (
                          <li key={r}>{label(r)}</li>
                        ))}
                      </ul>
                    </Detail>
                    <Detail title="Unused condition for change of mind">
                      <span className={item.policy.change_of_mind_requires_unused ? 'case-policy-req' : ''}>
                        {item.policy.change_of_mind_requires_unused ? 'Required' : 'Not required'}
                      </span>
                    </Detail>
                    <Detail title="Return required for refund">
                      <span className={item.policy.refund_requires_return ? 'case-policy-req' : ''}>
                        {item.policy.refund_requires_return ? 'Required' : 'Not required'}
                      </span>
                    </Detail>
                  </dl>

                  <p className="case-caption case-policy-description">{item.policy.description}</p>
                </>
              ) : (
                <>
                  <EmptyPanel
                    title={
                      pending === 'item' ? 'Loading item details\u2026'
                      : failure?.stage === 'item' ? 'Item details unavailable'
                      : 'Select an item above'
                    }
                    icon={
                      pending === 'item' ? (
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                          strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round">
                          <circle cx="12" cy="12" r="10" />
                          <path d="M12 6v6l4 2" />
                        </svg>
                      ) : undefined
                    }
                  >
                    {pending === 'item'
                      ? 'Fetching inventory and policy from the backend.'
                      : failure?.stage === 'item'
                        ? 'Check the error above before retrying.'
                        : 'Choose an ordered item to view its inventory level and applicable return policy.'}
                  </EmptyPanel>
                  {failure?.stage === 'item' && sku && (
                    <button
                      type="button"
                      className="button-secondary"
                      disabled={busy}
                      onClick={() => loadItem(sku)}
                    >
                      Retry item details
                    </button>
                  )}
                </>
              )}
            </section>
          </div>

          {/* Assessment grid — only shown once an item is loaded */}
          {item && (
            <div className="case-assessment-grid">

              {/* Assessment inputs */}
              <form
                className="case-panel"
                aria-labelledby="assessment-inputs-title"
                onSubmit={(event) => {
                  event.preventDefault(); setResult(null);
                  run('assessment', () => api.eligibility(context.order.id, {
                    customer_id: context.customer.id, sku, ...inputs,
                  }), setResult);
                }}
              >
                <div className="case-panel-heading">
                  <div>
                    <p className="eyebrow">Assessment inputs</p>
                    <h3 id="assessment-inputs-title">Scenario assumptions</h3>
                  </div>
                </div>

                <p className="case-caption case-assessment-note">
                  Enter the scenario inputs below. Delivery age, return reason, and item condition
                  are <strong>unverified</strong> — they have not been confirmed against order evidence
                  or payment history.
                </p>

                <fieldset disabled={busy}>
                  <div className="input-grid">
                    <label>
                      Requested quantity
                      <input
                        type="number" min="1" max="1000" step="1" required
                        value={inputs.quantity}
                        onChange={(event) => edit('quantity', event.target.value)}
                      />
                    </label>
                    <label>
                      Days since delivery
                      <input
                        type="number" min="0" max="36500" step="1" required
                        value={inputs.days_since_delivery}
                        onChange={(event) => edit('days_since_delivery', event.target.value)}
                      />
                    </label>
                    <label>
                      Return reason
                      <select
                        required
                        value={inputs.reason}
                        onChange={(event) => edit('reason', event.target.value)}
                      >
                        <option value="">Choose a reason\u2026</option>
                        {item.policy.allowed_reasons.map((value) => (
                          <option key={value} value={value}>{label(value)}</option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Item condition
                      <select
                        required
                        value={inputs.condition}
                        onChange={(event) => edit('condition', event.target.value)}
                      >
                        <option value="">Choose a condition\u2026</option>
                        {['unused', 'used', 'damaged'].map((value) => (
                          <option key={value} value={value}>{value}</option>
                        ))}
                      </select>
                    </label>
                  </div>

                  <div className="case-form-footer">
                    <button type="submit" disabled={busy} aria-busy={pending === 'assessment'}>
                      {pending === 'assessment' ? 'Checking\u2026' : 'Check eligibility'}
                    </button>
                    <span>Assessment only \u00b7 no action execution</span>
                  </div>
                </fieldset>
              </form>

              {/* Assessment result */}
              <section
                className={`case-panel${result ? ' assessment' : ''}`}
                aria-labelledby="assessment-result-title"
              >
                <div className="case-panel-heading">
                  <div>
                    <p className="eyebrow">Backend assessment</p>
                    <h3 id="assessment-result-title">Eligibility result</h3>
                  </div>
                  {result && (
                    <span className={`case-assessment-verdict ${result.return_eligible || result.refund_eligible ? 'case-assessment-verdict--partial' : 'case-assessment-verdict--denied'}`}>
                      {result.return_eligible || result.refund_eligible ? 'Conditionally eligible' : 'Ineligible'}
                    </span>
                  )}
                </div>

                {result ? (
                  <div role="status">
                    {/* Eligibility outcome tiles */}
                    <div className="eligibility-outcomes">
                      <div className={result.return_eligible ? 'eligibility-tile--eligible' : 'eligibility-tile--ineligible'}>
                        <span className="eligibility-tile-label">Return</span>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                          strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                          aria-hidden="true">
                          {result.return_eligible
                            ? <path d="M22 11.08V12a10 10 0 11-5.93-9.14M22 4L12 14.01l-3-3" />
                            : <><circle cx="12" cy="12" r="10" /><path d="M15 9l-6 6M9 9l6 6" /></>}
                        </svg>
                        <strong>{result.return_eligible ? 'Eligible' : 'Ineligible'}</strong>
                      </div>
                      <div className={result.refund_eligible ? 'eligibility-tile--eligible' : 'eligibility-tile--ineligible'}>
                        <span className="eligibility-tile-label">Refund</span>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                          strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                          aria-hidden="true">
                          {result.refund_eligible
                            ? <path d="M22 11.08V12a10 10 0 11-5.93-9.14M22 4L12 14.01l-3-3" />
                            : <><circle cx="12" cy="12" r="10" /><path d="M15 9l-6 6M9 9l6 6" /></>}
                        </svg>
                        <strong>{result.refund_eligible ? 'Eligible' : 'Ineligible'}</strong>
                      </div>
                    </div>

                    {/* Denial reasons */}
                    {result.denial_reasons.length > 0 && (
                      <div className="case-denials">
                        <h4>
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                            strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round"
                            aria-hidden="true">
                            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01" />
                          </svg>
                          Denial reasons
                        </h4>
                        <ul>
                          {result.denial_reasons.map((reason) => (
                            <li key={reason}>
                              {denialLabels[reason] || label(reason)}
                              {' '}<code>({reason})</code>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Operational boundary notice */}
                    <p className="case-boundary-notice">
                      Eligibility is not authorization. No action is executed by this workspace.
                    </p>

                    {/* Full evaluated inputs — collapsible for space efficiency */}
                    <details className="case-assessment-details">
                      <summary>Inputs evaluated by the API</summary>
                      <dl className="case-facts">
                        <Detail title="Order / Customer">
                          <span className="case-mono">{result.order_id}</span>
                          {' / '}
                          <span className="case-mono">{result.customer_id}</span>
                        </Detail>
                        <Detail title="SKU">
                          <span className="case-mono">{result.sku}</span>
                        </Detail>
                        <Detail title="Requested / Purchased quantity">
                          {result.requested_quantity} / {result.purchased_quantity}
                        </Detail>
                        <Detail title="Days since delivery">{result.days_since_delivery}</Detail>
                        <Detail title="Reason / Condition">
                          {label(result.reason)} / {label(result.condition)}
                        </Detail>
                        <Detail title="Policy">
                          <span className="case-mono">{result.policy_id}</span>
                          {' '}&middot; {result.policy_version}
                        </Detail>
                        <Detail title="Assessment basis">{label(result.assessment_basis)}</Detail>
                        <Detail title="Return required for refund">
                          {result.refund_requires_return ? 'Yes' : 'No'}
                        </Detail>
                        <Detail title="Authorization granted">
                          <span className={result.authorization_granted ? 'case-auth-yes' : 'case-auth-no'}>
                            {result.authorization_granted ? 'Yes' : 'No'}
                          </span>
                        </Detail>
                      </dl>
                    </details>
                  </div>
                ) : (
                  <EmptyPanel
                    title={
                      pending === 'assessment' ? 'Assessment in progress\u2026'
                      : failure?.stage === 'assessment' ? 'Assessment unavailable'
                      : 'No assessment yet'
                    }
                    icon={
                      pending === 'assessment' ? (
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                          strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round">
                          <circle cx="12" cy="12" r="10" />
                          <path d="M12 6v6l4 2" />
                        </svg>
                      ) : (
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                          strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round">
                          <path d="M9 4H5v17h14V4h-4M9 3h6v4H9zM8 13l3 3 5-6" />
                        </svg>
                      )
                    }
                  >
                    {pending === 'assessment'
                      ? 'Waiting for the backend eligibility result.'
                      : failure?.stage === 'assessment'
                        ? 'Check the error above before retrying.'
                        : 'Enter the scenario inputs and check eligibility to see the backend result here.'}
                  </EmptyPanel>
                )}
              </section>
            </div>
          )}
        </>
      )}
    </section>
  );
}
