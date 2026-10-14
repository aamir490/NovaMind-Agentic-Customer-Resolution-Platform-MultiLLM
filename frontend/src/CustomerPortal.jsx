import { useState } from 'react';
import CustomerCatalog from './CustomerCatalog.jsx';
import PlaceOrder from './PlaceOrder.jsx';
import MyOrders from './MyOrders.jsx';
import ReportProblem from './ReportProblem.jsx';
import MySupportRequests from './MySupportRequests.jsx';

/**
 * Customer Portal — top-level container for the customer-facing experience.
 *
 * Manages customer sub-views:
 *   catalog        → CustomerCatalog (browse products, select one to order)
 *   order          → PlaceOrder      (confirm & submit the order)
 *   my-orders      → MyOrders        (list + detail of past orders)
 *   report-problem → ReportProblem   (support intake form for a selected order)
 *   my-support-requests → MySupportRequests (read-only list and detail)
 *
 * The parent (main.jsx) renders this when role === 'CUSTOMER'.
 * Internal reviewer/admin views are NEVER rendered from here.
 *
 * Props
 * -----
 * api      — createApi() instance
 * identity — { user_id, role, customer_id, … }
 * key      — sessionVersion from parent resets internal state on sign-out
 */
export default function CustomerPortal({ api, identity }) {
  // Sub-view within the customer portal.
  const [portalView, setPortalView] = useState('catalog');
  // The catalog item the customer selected before navigating to order form.
  const [pendingItem, setPendingItem] = useState(null);
  // The order selected for the support intake form.
  const [pendingOrder, setPendingOrder] = useState(null);

  function handlePlaceOrder(item) {
    setPendingItem(item);
    setPortalView('order');
  }

  function handleOrderSuccess() {
    // After a successful order, navigate straight to My Orders.
    setPortalView('my-orders');
  }

  function handleBackToCatalog() {
    setPendingItem(null);
    setPortalView('catalog');
  }

  function handleReportProblem(order) {
    setPendingOrder(order);
    setPortalView('report-problem');
  }

  function handleBackToMyOrders() {
    setPendingOrder(null);
    setPortalView('my-orders');
  }

  return (
    <div className="customer-portal">
      {/* Portal sub-navigation */}
      <nav className="portal-subnav" aria-label="Customer portal navigation">
        <button
          type="button"
          className={`portal-subnav-item${portalView === 'catalog' || portalView === 'order' ? ' portal-subnav-item--active' : ''}`}
          aria-current={portalView === 'catalog' || portalView === 'order' ? 'page' : undefined}
          onClick={() => { setPendingItem(null); setPortalView('catalog'); }}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
            strokeLinejoin="round" strokeLinecap="round" aria-hidden="true">
            <path d="M6 2L3 6v14h18V6l-3-4zM3 6h18M16 10a4 4 0 01-8 0" />
          </svg>
          Catalog
        </button>
        <button
          type="button"
          className={`portal-subnav-item${portalView === 'my-orders' ? ' portal-subnav-item--active' : ''}`}
          aria-current={portalView === 'my-orders' ? 'page' : undefined}
          onClick={() => setPortalView('my-orders')}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
            strokeLinejoin="round" strokeLinecap="round" aria-hidden="true">
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          My Orders
        </button>
        <button
          type="button"
          className={`portal-subnav-item${portalView === 'my-support-requests' ? ' portal-subnav-item--active' : ''}`}
          aria-current={portalView === 'my-support-requests' ? 'page' : undefined}
          onClick={() => setPortalView('my-support-requests')}
        >
          My Support Requests
        </button>
      </nav>

      {/* Portal views — only one is shown at a time */}
      {portalView === 'catalog' && (
        <CustomerCatalog api={api} onPlaceOrder={handlePlaceOrder} />
      )}

      {portalView === 'order' && pendingItem && (
        <PlaceOrder
          api={api}
          item={pendingItem}
          onSuccess={handleOrderSuccess}
          onCancel={handleBackToCatalog}
        />
      )}

      {/* If somehow we land on 'order' without an item, fall back to catalog */}
      {portalView === 'order' && !pendingItem && (
        <CustomerCatalog api={api} onPlaceOrder={handlePlaceOrder} />
      )}

      {portalView === 'my-orders' && (
        <MyOrders api={api} onReportProblem={handleReportProblem} />
      )}

      {portalView === 'my-support-requests' && (
        <MySupportRequests api={api} />
      )}

      {portalView === 'report-problem' && pendingOrder && (
        <ReportProblem
          api={api}
          order={pendingOrder}
          onCancel={handleBackToMyOrders}
        />
      )}

      {/* If somehow we land on report-problem without an order, fall back */}
      {portalView === 'report-problem' && !pendingOrder && (
        <MyOrders api={api} onReportProblem={handleReportProblem} />
      )}
    </div>
  );
}
