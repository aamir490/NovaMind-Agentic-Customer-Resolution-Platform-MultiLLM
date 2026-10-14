import { useEffect, useState } from 'react';

function date(value) {
  return <time dateTime={value}>{new Date(value).toLocaleString()}</time>;
}

// A new selection mounts a fresh read, clearing old data and cancelling its request.
export default function MySupportRequests({ api }) {
  const [caseId, setCaseId] = useState(null);
  return <SupportRequestRead key={caseId ?? 'list'} api={api} caseId={caseId} onSelect={setCaseId} />;
}

function SupportRequestRead({ api, caseId, onSelect }) {
  const [state, setState] = useState({ phase: 'loading' });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setState({ phase: 'loading' });
    async function load() {
      try {
        const data = caseId
          ? await api.getMyCase(caseId, { signal: controller.signal })
          : await api.listMyCases({ signal: controller.signal });
        if (!controller.signal.aborted) setState({ phase: 'ready', data });
      } catch (error) {
        if (!controller.signal.aborted) setState({ phase: 'error',
          message: error.message || 'Could not load your support requests. Please try again.' });
      }
    }
    load();
    return () => controller.abort();
  }, [api, caseId, attempt]);

  function retry() {
    setState({ phase: 'loading' });
    setAttempt((value) => value + 1);
  }

  return (
    <section className="my-orders" aria-labelledby="my-support-requests-heading">
      <div className="orders-header">
        <h2 id="my-support-requests-heading">{caseId ? 'Support request details' : 'My Support Requests'}</h2>
        {caseId && <button type="button" className="button-secondary" onClick={() => onSelect(null)}>
          Back to My Support Requests
        </button>}
      </div>

      {state.phase === 'loading' && <p className="customer-notice" role="status">
        {caseId ? 'Loading your support request…' : 'Loading your support requests…'}
      </p>}

      {state.phase === 'error' && <div>
        <p className="customer-error" role="alert">{state.message}</p>
        <button type="button" className="button-secondary" onClick={retry}>Try again</button>
      </div>}

      {state.phase === 'ready' && (caseId ? (
        <div className="case-panel">
          <dl className="case-facts support-request-facts">
            <div><dt>Case ID</dt><dd className="order-id">{state.data.id}</dd></div>
            <div><dt>Order ID</dt><dd className="order-id">{state.data.order_id}</dd></div>
            <div><dt>Subject</dt><dd>{state.data.subject}</dd></div>
            <div><dt>Description</dt><dd className="support-request-description">{state.data.description}</dd></div>
            <div><dt>Status</dt><dd>{state.data.status}</dd></div>
            <div><dt>Created</dt><dd>{date(state.data.created_at)}</dd></div>
            <div><dt>Updated</dt><dd>{date(state.data.updated_at)}</dd></div>
          </dl>
        </div>
      ) : state.data.length === 0 ? (
        <div className="customer-empty" role="status">
          <h3>No support requests yet</h3>
          <p>You can report a problem from My Orders when you need help with an order.</p>
        </div>
      ) : (
        <ul className="orders-list" aria-label="Your support requests">
          {state.data.map((supportCase) => <li key={supportCase.id} className="order-row case-panel">
            <dl className="case-facts support-request-facts order-row-body">
              <div><dt>Subject</dt><dd>{supportCase.subject}</dd></div>
              <div><dt>Status</dt><dd>{supportCase.status}</dd></div>
              <div><dt>Created date</dt><dd>{date(supportCase.created_at)}</dd></div>
            </dl>
            <button type="button" className="button-secondary order-detail-btn"
              aria-label={`View details for ${supportCase.subject}`} onClick={() => onSelect(supportCase.id)}>
              View details
            </button>
          </li>)}
        </ul>
      ))}
    </section>
  );
}
