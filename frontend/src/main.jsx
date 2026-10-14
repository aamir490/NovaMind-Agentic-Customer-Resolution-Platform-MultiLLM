import { StrictMode, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';
import logoUrl from './assets/novamind-logo.png';
import Dashboard from './Dashboard.jsx';
import CaseReview from './CaseReview.jsx';
import AIWorkspace from './AIWorkspace.jsx';
import ReviewQueue from './ReviewQueue.jsx';
import AuditWorkspace from './AuditWorkspace.jsx';
import DiagnosticsWorkspace from './DiagnosticsWorkspace.jsx';
import CustomerPortal from './CustomerPortal.jsx';
import { createApi } from './api.js';

// Internal staff views — never shown to CUSTOMER role.
const staffViews = [
  { id: 'dashboard', label: 'Dashboard', title: 'Dashboard',
    description: 'Operations overview \u2014 cases, reviews, metrics and recent activity.' ,
    icon: 'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z' },
  { id: 'cases', label: 'Cases', title: 'Case review',
    description: 'Customer context, policy, and eligibility. One place to review the details.',
    icon: 'M4 5h16v15H4zM9 5V3h6v2M8 10h8M8 14h5' },
  { id: 'ai-workspace', label: 'AI Workspace', title: 'AI Workspace',
    description: 'Submit a case-bound request and follow its status and workflow events.',
    icon: 'M5 4h14v12H9l-4 4zM8 8h8M8 12h5' },
  { id: 'reviews', label: 'Reviews', title: 'Reviews',
    description: 'Assess pending proposals and record an explicit human review decision.',
    icon: 'M9 4H5v17h14V4h-4M9 3h6v4H9zM8 13l3 3 5-6' },
  { id: 'audit', label: 'Audit', title: 'Audit',
    description: 'Inspect safe workflow and human review metadata for an existing run.',
    icon: 'M5 3h10l4 4v14H5zM14 3v5h5M9 12h6M9 16h6' },
  { id: 'diagnostics', label: 'Diagnostics', title: 'Diagnostics',
    description: 'Inspect recorded operational metrics and recent trace metadata.',
    icon: 'M3 12h4l3-7 4 14 3-7h4' },
];

function App() {
  const [health, setHealth] = useState('idle');
  const [activeView, setActiveView] = useState('cases');
  const [session, setSession] = useState(null);
  const [sessionVersion, setSessionVersion] = useState(0);
  const [identityStatus, setIdentityStatus] = useState('signed-out');
  const [identityMessage, setIdentityMessage] = useState('Enter your existing local access token to confirm your identity.');
  const identityClient = useRef(null);

  // Derive whether this session is a customer; drives what is shown/hidden.
  const isCustomer = session?.identity?.role === 'CUSTOMER';

  // Staff view state — only relevant when not a customer.
  const view = staffViews.find((entry) => entry.id === activeView) ?? staffViews[0];

  function clearIdentity() {
    identityClient.current?.dispose();
    identityClient.current = null;
    setSession(null);
    // Drop case data across identities; ordinary view navigation still preserves it.
    setSessionVersion((version) => version + 1);
  }

  function signOut() {
    clearIdentity();
    setIdentityStatus('signed-out');
    setIdentityMessage('Signed out locally. Credentials and workspace data have been cleared. Submitted runs are not cancelled.');
  }

  async function confirmIdentity(event) {
    event.preventDefault();
    if (identityStatus === 'checking') return;
    const form = event.currentTarget;
    const token = new FormData(form).get('token');
    form.reset();
    clearIdentity();
    setIdentityStatus('checking');
    setIdentityMessage('Confirming your identity with the local API…');
    const candidate = createApi(globalThis.fetch, {
      token,
      onUnauthorized() {
        if (identityClient.current !== candidate) return;
        clearIdentity();
        setIdentityStatus('error');
        setIdentityMessage('Authentication required. Enter a valid token configured for the local API.');
      },
    });
    identityClient.current = candidate;
    try {
      const identity = await candidate.identity();
      // Ignore responses from a cancelled or replaced identity request.
      if (identityClient.current !== candidate) return;
      setSession({ identity, api: candidate });
      setSessionVersion((version) => version + 1);
      setIdentityStatus('authenticated');
      setIdentityMessage('Identity confirmed by the local API.');
    } catch (error) {
      if (identityClient.current !== candidate) return;
      clearIdentity();
      setIdentityStatus('error');
      setIdentityMessage(error.status === 403
        ? 'Access denied by the local API.'
        : 'Identity could not be confirmed. Check the local API and try again.');
    }
  }

  async function checkHealth() {
    setHealth('checking');
    try {
      const response = await fetch('/api/health', {
        signal: AbortSignal.timeout(5000),
        cache: 'no-store',
      });
      if (!response.ok) throw new Error('API request failed');
      const data = await response.json();
      if (data.status !== 'ok' || data.service !== 'novamind-api') {
        throw new Error('Unexpected health response');
      }
      setHealth('online');
    } catch {
      setHealth('offline');
    }
  }

  const messages = {
    idle: 'Connection not checked. Check the API when you\'re ready to load case data.',
    checking: 'Checking the local API\u2026',
    online: 'The local API is responding. This check confirms connectivity only.',
    offline: 'The local API is unavailable. Check that the backend is running, then try again.',
  };

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">Skip to workspace</a>

      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      <aside className="sidebar" aria-label="Workspace navigation">

        {/* Brand / logo — vertical stack: logo → title → subtitle */}
        <a className="brand" href="#workspace" aria-label="NovaMind AI — go to workspace">
          <img
            src={logoUrl}
            alt="NovaMind AI logo"
            className="brand-logo"
            width="130"
            height="65"
          />
          <span className="brand-text">
            NovaMind <span className="brand-ai">AI</span>
            <span className="brand-caption">Customer resolution</span>
          </span>
        </a>

        {/* Navigation */}
        <nav className="workspace-nav" aria-label="Main navigation">
          {isCustomer ? (
            <>
              <p className="nav-label">My account</p>
              <div className="nav-items">
                <span className="nav-item nav-item--customer-label">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
                    strokeLinejoin="round" strokeLinecap="round" aria-hidden="true">
                    <circle cx="12" cy="8" r="4" />
                    <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
                  </svg>
                  Customer portal
                </span>
              </div>
            </>
          ) : (
            <>
              <p className="nav-label">Workspace</p>
              <div className="nav-items">
                {staffViews.map((entry) => (
                  <button key={entry.id} type="button" className="nav-item"
                    aria-current={activeView === entry.id ? 'page' : undefined}
                    aria-controls="workspace"
                    onClick={() => setActiveView(entry.id)}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
                      strokeLinejoin="round" strokeLinecap="round" aria-hidden="true">
                      <path d={entry.icon} />
                    </svg>
                    {entry.label}
                    {activeView === entry.id && (
                      <span className="nav-indicator" aria-hidden="true" />
                    )}
                  </button>
                ))}
              </div>
            </>
          )}
        </nav>

        {/* Sidebar footer — context + attribution + social links */}
        <div className="sidebar-footer">
          {/* Current workspace context */}
          <div className="sidebar-context">
            <span className="context-symbol" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
                strokeLinejoin="round" strokeLinecap="round" aria-hidden="true">
                <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 4v4l3 3" />
              </svg>
            </span>
            <div>
              <strong>Local workspace</strong>
              <span>Customer support</span>
            </div>
          </div>

          {/* Attribution + social links */}
          <div className="sidebar-attribution">
            <span className="sidebar-built-by">Built by Aamir</span>
            <div className="sidebar-social">
              <a
                href="https://github.com/aamir490"
                target="_blank"
                rel="noopener noreferrer"
                className="social-link"
                aria-label="Aamir on GitHub">
                {/* GitHub icon */}
                <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <path d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.483 0-.237-.009-.868-.013-1.703-2.782.605-3.369-1.342-3.369-1.342-.454-1.154-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0 1 12 6.836a9.59 9.59 0 0 1 2.504.337c1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.202 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.579.688.481C19.138 20.163 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
                </svg>
              </a>
              <a
                href="https://www.linkedin.com/in/aamir-imran"
                target="_blank"
                rel="noopener noreferrer"
                className="social-link"
                aria-label="Aamir on LinkedIn">
                {/* LinkedIn icon */}
                <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
              </a>
            </div>
          </div>
        </div>
      </aside>

      {/* ── App body ────────────────────────────────────────────────── */}
      <div className="app-body">

        {/* Topbar */}
        <header className="topbar">
          <p className="breadcrumb">
            <span>Workspace</span>
            <span aria-hidden="true">/</span>
            {isCustomer ? 'Customer portal' : view.label}
          </p>
          <div className="topbar-end">
            <span className="environment-label">Local environment</span>
          </div>
        </header>

        {/* Main workspace */}
        <main id="workspace" className="workspace" tabIndex={-1}>

          {/* Identity panel */}
          <section className="identity-panel" aria-labelledby="identity-title">
            <div className="identity-heading">
              <h2 id="identity-title">Workspace identity</h2>
              <span className={`badge ${session ? 'identity-role' : ''}`}>
                {session
                  ? session.identity.role
                  : identityStatus === 'checking' ? 'Verifying' : 'Not signed in'}
              </span>
            </div>
            <p id="identity-status" className={`identity-status ${identityStatus}`}
              role="status" aria-live="polite">
              {identityMessage}
            </p>
            {session
              ? <div className="identity-details">
                  <dl>
                    <dt>User ID</dt><dd>{session.identity.user_id}</dd>
                    {session.identity.customer_id && (
                      <><dt>Customer ID</dt><dd>{session.identity.customer_id}</dd></>
                    )}
                  </dl>
                  <button type="button" className="button-secondary" onClick={signOut}>
                    Sign out
                  </button>
                </div>
              : <form className="identity-form" onSubmit={confirmIdentity} autoComplete="off">
                  <label htmlFor="local-access-token">Local access token
                    <input id="local-access-token" name="token" type="password"
                      required minLength={16} maxLength={256}
                      autoComplete="off" spellCheck={false} autoCapitalize="none"
                      disabled={identityStatus === 'checking'}
                      aria-describedby="identity-help identity-status" />
                  </label>
                  <div className="identity-actions">
                    <button type="submit" disabled={identityStatus === 'checking'}>
                      {identityStatus === 'checking' ? 'Verifying\u2026' : 'Confirm identity'}
                    </button>
                    {identityStatus === 'checking' && (
                      <button type="button" className="button-secondary" onClick={signOut}>
                        Cancel
                      </button>
                    )}
                  </div>
                  <p id="identity-help" className="identity-help">
                    Kept in memory for this page only. Reloading or signing out clears it.
                  </p>
                </form>
            }
          </section>

          {/* Page heading */}
          <div className="page-heading">
            <div>
              <p className="eyebrow">{isCustomer ? 'NovaMind AI' : 'Customer resolution'}</p>
              <h1 aria-live="polite">{isCustomer ? 'Customer portal' : view.title}</h1>
              <p className="intro">
                {isCustomer
                  ? 'Browse products, place a demo order, and track your orders.'
                  : view.description}
              </p>
            </div>
            <span className="badge">
              {isCustomer
                ? 'Customer portal'
                : ['cases', 'audit', 'diagnostics'].includes(activeView)
                  ? 'Read-only workspace'
                  : activeView === 'reviews'
                    ? 'Human review'
                    : activeView === 'ai-workspace'
                      ? 'Live run tracking'
                      : activeView === 'dashboard'
                        ? 'Operations overview'
                        : 'Placeholder view'}
            </span>
          </div>

          {/* Customer portal — shown only to CUSTOMER role */}
          {isCustomer && (
            <CustomerPortal key={sessionVersion} api={session.api} identity={session.identity} />
          )}

          {/* Staff workspace — shown only to non-CUSTOMER roles */}
          {!isCustomer && <>
            {view.placeholder && (
              <section className="view-placeholder" aria-labelledby="placeholder-title">
                <p className="eyebrow">{view.label}</p>
                <h2 id="placeholder-title">This view is not connected yet.</h2>
                <p>{view.placeholder}</p>
              </section>
            )}
            {activeView === 'dashboard' && (
              <Dashboard
                key={sessionVersion}
                api={session?.api}
                identity={session?.identity}
                onNavigate={setActiveView}
              />
            )}
            <div hidden={activeView !== 'ai-workspace'}>
              <AIWorkspace key={sessionVersion} api={session?.api} identity={session?.identity} />
            </div>
            <div hidden={activeView !== 'reviews'}>
              <ReviewQueue key={sessionVersion} api={session?.api} identity={session?.identity}
                active={activeView === 'reviews'} />
            </div>
            {activeView === 'audit' && (
              <AuditWorkspace key={sessionVersion} api={session?.api} identity={session?.identity} />
            )}
            {activeView === 'diagnostics' && (
              <DiagnosticsWorkspace key={sessionVersion} api={session?.api} identity={session?.identity} />
            )}
            {/* Keep the case workspace mounted so navigation preserves an in-progress review. */}
            <div className="workspace-grid cases-layout" hidden={activeView !== 'cases'}>
              <CaseReview key={sessionVersion} api={session?.api} />
              <aside className="workspace-rail" aria-label="Workspace information">
                <section className="connection-panel" aria-labelledby="connection-title">
                  <div className="panel-heading">
                    <span className="panel-icon" aria-hidden="true">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <rect x="4" y="4" width="16" height="6" rx="2" />
                        <rect x="4" y="14" width="16" height="6" rx="2" />
                        <path d="M8 7h.01M8 17h.01M12 7h5M12 17h5" strokeLinecap="round" />
                      </svg>
                    </span>
                    <div>
                      <p className="eyebrow">Connection</p>
                      <h2 id="connection-title">Local API</h2>
                    </div>
                  </div>
                  <p className={`status ${health}`} role="status" aria-live="polite">
                    {messages[health]}
                  </p>
                  <button className="button-secondary" onClick={checkHealth}
                    disabled={health === 'checking'}>
                    {health === 'checking' ? 'Checking\u2026' : 'Check local API'}
                  </button>
                </section>
                <section className="guidance-panel" aria-labelledby="boundary-title">
                  <p className="eyebrow">Review boundary</p>
                  <h2 id="boundary-title">Evidence before action.</h2>
                  <p>Eligibility is not authorization. Approval is not execution.</p>
                  <span className="guidance-rule" aria-hidden="true" />
                  <p className="footnote">
                    This workspace supports read-only assessment. No business action is performed here.
                  </p>
                </section>
              </aside>
            </div>
          </>}

          {/* Workspace footer */}
          <footer className="workspace-footer">
            <span>
              NovaMind AI
              <span aria-hidden="true"> / </span>
              Customer Resolution Platform
            </span>
            <span className="workspace-footer-links">
              <a href="https://github.com/aamir490"
                target="_blank" rel="noopener noreferrer"
                className="footer-link">
                GitHub
              </a>
              <span aria-hidden="true">&middot;</span>
              <a href="https://www.linkedin.com/in/aamir-imran"
                target="_blank" rel="noopener noreferrer"
                className="footer-link">
                LinkedIn
              </a>
            </span>
            <span>
              {isCustomer
                ? 'Customer portal'
                : activeView === 'cases'
                  ? 'Local case review'
                  : 'Local workspace'}
            </span>
          </footer>

        </main>
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>);
