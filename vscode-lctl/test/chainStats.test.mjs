import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { test } from 'node:test';

const require = createRequire(import.meta.url);
const { chainTokens, chainCost, limitEvents, MODEL_PRICING, pricingFor } = require('../out/chainStats.js');

const ev = (seq, type, data) => ({ seq, type, timestamp: 't', data });

test('chainTokens reads the library shape data.tokens.{input,output}', () => {
  const events = [
    ev(1, 'step_end', { tokens: { input: 10, output: 5 } }),
    ev(2, 'step_end', { tokens: { input: 1, output: 2 } }),
  ];
  assert.deepEqual(chainTokens(events), { input: 11, output: 7 });
});

test('chainTokens falls back to legacy tokens_in/tokens_out', () => {
  assert.deepEqual(chainTokens([ev(1, 'step_end', { tokens_in: 7, tokens_out: 3 })]), { input: 7, output: 3 });
});

test('chainTokens tolerates events with no tokens, no data, or junk', () => {
  const events = [
    ev(1, 'step_end', { duration_ms: 5 }),
    { seq: 2, type: 'step_end', timestamp: 't' },
    ev(3, 'step_end', { tokens: { input: 'x', output: null } }),
    ev(4, 'tool_call', { tokens: { input: 99, output: 99 } }),
  ];
  assert.deepEqual(chainTokens(events), { input: 0, output: 0 });
  assert.deepEqual(chainTokens([]), { input: 0, output: 0 });
});

test('chainTokens is non-zero on a trace written by the Python library', () => {
  const chain = JSON.parse(readFileSync(new URL('./fixtures/library-trace.lctl.json', import.meta.url), 'utf8'));
  assert.deepEqual(chainTokens(chain.events), { input: 3500, output: 1000 });
});

test('chainCost uses the given per-1M pricing', () => {
  assert.equal(chainCost({ input: 1_000_000, output: 2_000_000 }, { input: 3, output: 15 }), 33);
  assert.equal(chainCost({ input: 500_000, output: 0 }, { input: 10, output: 30 }), 5);
  assert.equal(chainCost({ input: 0, output: 0 }, { input: 10, output: 30 }), 0);
});

test('pricingFor resolves a model name and falls back to default', () => {
  assert.deepEqual(pricingFor('gpt-4'), MODEL_PRICING['gpt-4']);
  assert.deepEqual(pricingFor('nope'), MODEL_PRICING['default']);
  assert.deepEqual(pricingFor(undefined), MODEL_PRICING['default']);
});

test('limitEvents respects max and bad values', () => {
  const events = [1, 2, 3, 4].map((n) => ev(n, 'step_end', {}));
  assert.equal(limitEvents(events, 2).length, 2);
  assert.equal(limitEvents(events, 2)[1].seq, 2);
  assert.equal(limitEvents(events, 100).length, 4);
  assert.equal(limitEvents(events, 0).length, 0);
  assert.equal(limitEvents(events, -5).length, 0);
  assert.equal(limitEvents(events, Number.NaN).length, 4);
});
