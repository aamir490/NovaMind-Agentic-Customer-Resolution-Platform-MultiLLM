import test from 'node:test';
import assert from 'node:assert/strict';

import { formatCatalogPrice, selectCatalogProducts } from '../frontend/src/catalogView.js';

function product({ sku, name, description = 'Catalog description', amount = '10.00', available = true } = {}) {
  return {
    id: `10000000-0000-4000-8000-${sku.padStart(12, '0')}`,
    sku, name, description, available,
    unit_price: { amount, currency: 'USD' },
  };
}

const products = [
  product({ sku: 'LAP-2', name: 'Compact Laptop', description: 'Portable work computer', amount: '999.99', available: false }),
  product({ sku: 'ACC-1', name: 'USB-C Dock', description: 'Display and network connections', amount: '149.99' }),
  product({ sku: 'BAG-1', name: 'Laptop Sleeve', description: 'Water-resistant protection', amount: '29.99' }),
];

test('catalog search uses only backend-provided name, SKU, and description fields', () => {
  assert.deepEqual(selectCatalogProducts(products, { query: 'usb-c' }).map((item) => item.sku), ['ACC-1']);
  assert.deepEqual(selectCatalogProducts(products, { query: 'portable WORK' }).map((item) => item.sku), ['LAP-2']);
  assert.deepEqual(selectCatalogProducts(products, { query: 'bag-1' }).map((item) => item.sku), ['BAG-1']);
});

test('availability filtering preserves the backend boolean without inventing stock counts', () => {
  assert.deepEqual(selectCatalogProducts(products, { availability: 'available' }).map((item) => item.sku), ['BAG-1', 'ACC-1']);
  assert.deepEqual(selectCatalogProducts(products, { availability: 'unavailable' }).map((item) => item.sku), ['LAP-2']);
});

test('catalog sorting supports name and exact decimal price ordering', () => {
  assert.deepEqual(selectCatalogProducts(products, { sort: 'name-desc' }).map((item) => item.sku), ['ACC-1', 'BAG-1', 'LAP-2']);
  assert.deepEqual(selectCatalogProducts(products, { sort: 'price-asc' }).map((item) => item.sku), ['BAG-1', 'ACC-1', 'LAP-2']);
  assert.deepEqual(selectCatalogProducts(products, { sort: 'price-desc' }).map((item) => item.sku), ['LAP-2', 'ACC-1', 'BAG-1']);
});

test('featured order remains available-first and does not mutate API results', () => {
  const original = [...products];
  const selected = selectCatalogProducts(products);
  assert.deepEqual(selected.map((item) => item.sku), ['BAG-1', 'ACC-1', 'LAP-2']);
  assert.deepEqual(products, original);
  assert.equal(selected[0], products[2]);
});

test('catalog price formatting uses the supplied amount and currency', () => {
  assert.equal(formatCatalogPrice({ amount: '1299.99', currency: 'USD' }, 'en-US'), '$1,299.99');
  assert.equal(formatCatalogPrice({ amount: 'not-a-number', currency: 'USD' }, 'en-US'), 'USD not-a-number');
});
