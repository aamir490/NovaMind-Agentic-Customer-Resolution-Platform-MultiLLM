import { useEffect, useRef, useState } from 'react';

/**
 * Product catalog browser for authenticated customers.
 *
 * Props
 * -----
 * api          — createApi() instance (required)
 * onPlaceOrder — called with { sku, name, quantity } when customer clicks
 *                "Add to order" so the parent can navigate to the order form
 */
export default function CustomerCatalog({ api, onPlaceOrder }) {
  const [products, setProducts] = useState(null);   // null = not loaded
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // Per-product quantity inputs, keyed by sku.
  const [quantities, setQuantities] = useState({});
  const requestRef = useRef(0);

  useEffect(() => {
    // Reload whenever the api instance changes (new identity).
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  function load() {
    const version = ++requestRef.current;
    setLoading(true);
    setError(null);
    api.listCatalog()
      .then((data) => {
        if (requestRef.current !== version) return;
        setProducts(data);
        // Seed quantities to 1 for every available product.
        const init = {};
        data.forEach((p) => { if (p.available) init[p.sku] = 1; });
        setQuantities(init);
      })
      .catch((err) => {
        if (requestRef.current !== version) return;
        setError(err.message || 'Could not load catalog. Check the API and try again.');
      })
      .finally(() => { if (requestRef.current === version) setLoading(false); });
  }

  function handleQtyChange(sku, raw) {
    const value = Math.max(1, Math.min(1000, parseInt(raw, 10) || 1));
    setQuantities((prev) => ({ ...prev, [sku]: value }));
  }

  function handleAddToOrder(product) {
    const quantity = quantities[product.sku] ?? 1;
    onPlaceOrder?.({ sku: product.sku, name: product.name, quantity });
  }

  // Stable sort: available first, then alphabetically by name.
  const sorted = products
    ? [...products].sort((a, b) => {
        if (a.available !== b.available) return a.available ? -1 : 1;
        return a.name.localeCompare(b.name);
      })
    : [];

  return (
    <section className="customer-catalog" aria-labelledby="catalog-heading">
      <div className="catalog-header">
        <div>
          <p className="eyebrow">Product catalog</p>
          <h2 id="catalog-heading">Browse products</h2>
        </div>
        <button
          type="button"
          className="button-secondary"
          onClick={load}
          disabled={loading}
          aria-busy={loading}
        >
          {loading ? 'Loading…' : 'Refresh catalog'}
        </button>
      </div>

      {loading && (
        <p className="customer-notice" role="status" aria-live="polite">
          Loading product catalog…
        </p>
      )}

      {error && (
        <p className="customer-error" role="alert">{error}</p>
      )}

      {!loading && !error && products === null && (
        <div className="customer-empty" role="status">
          <span className="customer-empty-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
              strokeLinejoin="round" strokeLinecap="round">
              <path d="M6 2L3 6v14h18V6l-3-4zM3 6h18M16 10a4 4 0 01-8 0" />
            </svg>
          </span>
          <h3>Catalog not loaded</h3>
          <p>Click "Refresh catalog" to load available products.</p>
        </div>
      )}

      {!loading && products !== null && products.length === 0 && (
        <div className="customer-empty" role="status">
          <h3>No products available</h3>
          <p>The catalog returned no products. Try again later.</p>
        </div>
      )}

      {sorted.length > 0 && (
        <ul className="catalog-grid" aria-label="Product list">
          {sorted.map((product) => (
            <li key={product.id} className={`catalog-card${product.available ? '' : ' catalog-card--unavailable'}`}>
              <div className="catalog-card-body">
                <div className="catalog-card-meta">
                  <span className="catalog-sku">{product.sku}</span>
                  {product.available
                    ? <span className="catalog-badge catalog-badge--available">Available</span>
                    : <span className="catalog-badge catalog-badge--unavailable">Out of stock</span>}
                </div>
                <h3 className="catalog-card-name">{product.name}</h3>
                <p className="catalog-card-description">{product.description}</p>
                <p className="catalog-card-price" aria-label={`Price: ${product.unit_price.amount} ${product.unit_price.currency}`}>
                  <strong>{product.unit_price.currency} {product.unit_price.amount}</strong>
                </p>
              </div>
              {product.available && (
                <div className="catalog-card-action">
                  <label className="catalog-qty-label">
                    <span>Quantity</span>
                    <input
                      type="number"
                      min="1"
                      max="1000"
                      step="1"
                      value={quantities[product.sku] ?? 1}
                      aria-label={`Quantity for ${product.name}`}
                      onChange={(e) => handleQtyChange(product.sku, e.target.value)}
                    />
                  </label>
                  <button
                    type="button"
                    onClick={() => handleAddToOrder(product)}
                    aria-label={`Place order for ${product.name}`}
                  >
                    Place order
                  </button>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
