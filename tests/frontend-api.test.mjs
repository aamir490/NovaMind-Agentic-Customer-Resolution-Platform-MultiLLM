import test from 'node:test';
import assert from 'node:assert/strict';
import { createApi } from '../frontend/src/api.js';

test('case context uses existing IDs and GET requests only', async () => {
  const calls = [];
  const values = [{ customer_id: 'customer-1', order_id: 'order-1' }, { name: 'Demo' }, { items: [] }];
  const api = createApi(async (url, options) => {
    calls.push([url, options]);
    return { ok: true, json: async () => values.shift() };
  });
  const result = await api.context('case-1');
  assert.equal(result.customer.name, 'Demo');
  assert.deepEqual(calls.map(([url]) => url), ['/api/cases/case-1', '/api/customers/customer-1', '/api/orders/order-1']);
  assert.ok(calls.every(([, options]) => options.method === 'GET' && options.body === undefined));
});

test('inventory chooses the returned policy, not a hard-coded policy', async () => {
  const calls = [];
  const api = createApi(async (url) => {
    calls.push(url);
    return { ok: true, json: async () => calls.length === 1
      ? { sku: 'LAP /1', policy_id: 'policy/demo', available_quantity: 0 }
      : { id: 'policy/demo', version: 'v1', return_window_days: 30, allowed_reasons: ['wrong_item'],
        change_of_mind_requires_unused: true, refund_requires_return: true, description: 'Local policy' } };
  });
  const result = await api.item('LAP /1');
  assert.deepEqual(calls, ['/api/inventory/LAP%20%2F1', '/api/policies/policy%2Fdemo']);
  assert.equal(result.inventory.available_quantity, 0);
});

test('eligibility preserves inputs and backend decisions without frontend rules', async () => {
  const inputs = { customer_id: 'c', sku: 'A&B', quantity: '2', days_since_delivery: '31', reason: 'wrong_item', condition: 'used' };
  const decision = { order_id: 'order', customer_id: inputs.customer_id, sku: inputs.sku,
    reason: inputs.reason, condition: inputs.condition, requested_quantity: 2, days_since_delivery: 31,
    purchased_quantity: 2, policy_id: 'policy/demo', policy_version: 'v1', return_eligible: false,
    refund_eligible: false, refund_requires_return: true, denial_reasons: ['new_backend_reason'],
    assessment_basis: 'local_scenario_unverified_inputs', authorization_granted: false };
  const api = createApi(async (url, options) => {
    assert.equal(options.method, 'GET');
    assert.deepEqual(Object.fromEntries(new URL(url, 'http://localhost').searchParams), inputs);
    return { ok: true, json: async () => decision };
  });
  assert.deepEqual(await api.eligibility('order', inputs), decision);
});

test('missing inventory rejects rather than inventing zero stock', async () => {
  let calls = 0;
  const api = createApi(async () => { calls++; return { ok: false, status: 404, json: async () => ({ detail: 'Inventory item not found' }) }; });
  await assert.rejects(api.item('unknown'), /Inventory item not found/);
  assert.equal(calls, 1);
});

test('validation errors and non-JSON errors are readable', async () => {
  const api = createApi(async () => ({ ok: false, status: 422, json: async () => ({ detail: [{ msg: 'Invalid quantity' }] }) }));
  await assert.rejects(api.listCases(), /Invalid quantity/);
  const unavailable = createApi(async () => ({ ok: false, status: 502, json: async () => { throw new Error(); } }));
  await assert.rejects(unavailable.listCases(), /502/);
});

test('empty case list is preserved and network failure is not hidden', async () => {
  assert.deepEqual(await createApi(async () => ({ ok: true, json: async () => [] })).listCases(), []);
  await assert.rejects(createApi(async () => { throw new Error('Offline'); }).listCases(), /Offline/);
});
