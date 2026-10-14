import { useState } from 'react';

/**
 * Order placement form.
 *
 * Receives a pre-filled item (sku, name, quantity) from the catalog selection.
 * Customers can adjust quantity before submitting.
 *
 * SECURITY: customer_id is NEVER included in the request body.
 * The backend derives it exclusively from the authenticated identity.
 *
 * Props
 * -----
 * api          — createApi() instance
 * item         — { sku, name, quantity } selected from the catalog
 * onSuccess    — called with the created Order object on success
 * onCancel     — called when the customer clicks "Back to catalog"
 */
export default function PlaceOrder({ api, item, onSuccess, onCancel }) {
  const [quantity, setQuantity] = useState(item?.quantity ?? 1);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);   // prevent duplicate POST
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);

  // Guard against stale item prop — reset form if item changes.
  const [lastSku, setLastSku] = useState(item?.sku);
  if (item?.sku !== lastSku) {
    setLastSku(item?.sku);
    setQuantity(item?.quantity ?? 1);
    setResult(null);
    setError(null);
    setSubmitting(false);
    setSubmitted(false);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    // Strict single-submission guard: once submitted, block all further clicks
    // until the response arrives and we reset, or the user goes back.
    if (submitting || submitted) return;

    const qty = Math.max(1, Math.min(1000, parseInt(quantity, 10) || 1));
    setQuantity(qty);
    setSubmitting(true);
    setSubmitted(true);
    setError(null);

    try {
      // No customer_id field — the backend owns it.
      const order = await api.createMyOrder([{ sku: item.sku, quantity: qty }]);
      setResult(order);
      onSuccess?.(order);
    } catch (err) {
      setError(err.message || 'Order could not be placed. Please try again.');
      // Allow resubmit after an error by clearing the submitted guard.
      setSubmitted(false);
    } finally {
      setSubmitting(false);
    }
  }

  // Success state
  if (result) {
    return (
      <section className="place-order" aria-labelledby="order-success-heading">
        <div className="order-success" role="status" aria-live="polite">
          <span className="order-success-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
              strokeLinejoin="round" strokeLinecap="round">
              <path d="M22 11.08V12a10 10 0 11-5.93-9.14M22 4L12 14.01l-3-3" />
            </svg>
          </span>
          <h2 id="order-success-heading">Order placed</h2>
          <p>Your order has been received. Here's a summary:</p>
          <dl className="case-facts order-summary-facts">
            <div><dt>Order ID</dt><dd className="order-id">{result.id}</dd></div>
            {result.items.map((it, i) => (
              <div key={i}>
                <dt>{it.name}</dt>
                <dd>SKU: {it.sku} · Qty: {it.quantity}</dd>
              </div>
            ))}
          </dl>
          <div className="order-success-actions">
            <button type="button" onClick={onCancel}>Back to catalog</button>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="place-order" aria-labelledby="place-order-heading">
      <div className="order-form-header">
        <div>
          <p className="eyebrow">Place order</p>
          <h2 id="place-order-heading">Confirm your order</h2>
        </div>
        <button type="button" className="button-secondary" onClick={onCancel}>
          ← Back to catalog
        </button>
      </div>

      <div className="order-product-summary case-panel">
        <div className="catalog-card-meta">
          <span className="catalog-sku">{item?.sku}</span>
        </div>
        <h3 className="catalog-card-name">{item?.name}</h3>
        <p className="case-caption">
          Review the quantity below and click "Place demo order" to submit.
        </p>
      </div>

      <form className="order-form case-panel" onSubmit={handleSubmit} noValidate>
        <fieldset disabled={submitting}>
          <label>
            Quantity
            <input
              type="number"
              min="1"
              max="1000"
              step="1"
              required
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              aria-describedby="qty-help"
            />
          </label>
          <p id="qty-help" className="case-caption">Enter a quantity between 1 and 1000.</p>

          {/* Explicit note: customer_id is NOT sent — this is intentional */}
          <p className="order-ownership-note case-caption">
            Your account is identified automatically. No personal ID is transmitted in this request.
          </p>
        </fieldset>

        {error && (
          <p className="customer-error" role="alert">{error}</p>
        )}

        <div className="order-form-actions">
          <button
            type="submit"
            disabled={submitting || submitted}
            aria-busy={submitting}
          >
            {submitting ? 'Placing order…' : 'Place demo order'}
          </button>
          <span className="case-caption">
            {submitting ? 'Your order is being submitted…' : 'Demo order · no real transaction'}
          </span>
        </div>
      </form>
    </section>
  );
}
