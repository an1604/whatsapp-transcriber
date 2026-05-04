'use strict';

const { test, describe, before, after, mock } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');

const { createHttpClient, DEFAULT_TIMEOUT_MS } = require('../src/pipeline_client');

// ---------------------------------------------------------------------------
// Helpers: minimal in-process HTTP server to avoid real network calls
// ---------------------------------------------------------------------------

function makeServer(handler) {
  return new Promise((resolve) => {
    const server = http.createServer(handler);
    server.listen(0, '127.0.0.1', () => {
      resolve(server);
    });
  });
}

function serverUrl(server) {
  const { port } = server.address();
  return `http://127.0.0.1:${port}`;
}

// ---------------------------------------------------------------------------
// submitUrl
// ---------------------------------------------------------------------------
describe('createHttpClient — submitUrl', () => {
  let server;
  let client;
  let lastRequest;

  before(async () => {
    server = await makeServer((req, res) => {
      let body = '';
      req.on('data', (chunk) => (body += chunk));
      req.on('end', () => {
        lastRequest = { method: req.method, url: req.url, body };
        if (req.url === '/api/submit' && req.method === 'POST') {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(
            JSON.stringify({
              video_id: 42,
              canonical_url: 'https://www.youtube.com/watch?v=abc',
              platform: 'youtube',
              is_duplicate: false,
            })
          );
        } else {
          res.writeHead(404);
          res.end('not found');
        }
      });
    });
    client = createHttpClient(serverUrl(server));
  });

  after(async () => {
    await new Promise((r) => server.close(r));
  });

  test('sends POST to /api/submit', async () => {
    await client.submitUrl('https://youtu.be/abc123');
    assert.equal(lastRequest.method, 'POST');
    assert.equal(lastRequest.url, '/api/submit');
  });

  test('sends url in JSON body', async () => {
    await client.submitUrl('https://youtu.be/abc123');
    const parsed = JSON.parse(lastRequest.body);
    assert.equal(parsed.url, 'https://youtu.be/abc123');
  });

  test('sends source_group in JSON body when provided', async () => {
    await client.submitUrl('https://youtu.be/abc123', 'group@g.us');
    const parsed = JSON.parse(lastRequest.body);
    assert.equal(parsed.source_group, 'group@g.us');
  });

  test('source_group is null when omitted', async () => {
    await client.submitUrl('https://youtu.be/abc123');
    const parsed = JSON.parse(lastRequest.body);
    assert.equal(parsed.source_group, null);
  });

  test('returns parsed JSON response', async () => {
    const result = await client.submitUrl('https://youtu.be/abc123');
    assert.equal(result.video_id, 42);
    assert.equal(result.platform, 'youtube');
    assert.equal(result.is_duplicate, false);
  });

  test('throws TypeError for empty url', async () => {
    await assert.rejects(() => client.submitUrl(''), TypeError);
  });

  test('throws TypeError for non-string url', async () => {
    await assert.rejects(() => client.submitUrl(null), TypeError);
  });
});

// ---------------------------------------------------------------------------
// submitUrl — error responses
// ---------------------------------------------------------------------------
describe('createHttpClient — submitUrl error handling', () => {
  let server;
  let client;

  before(async () => {
    server = await makeServer((_req, res) => {
      res.writeHead(422, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ detail: 'Unsupported platform' }));
    });
    client = createHttpClient(serverUrl(server));
  });

  after(async () => {
    await new Promise((r) => server.close(r));
  });

  test('throws Error on non-2xx response', async () => {
    await assert.rejects(
      () => client.submitUrl('https://youtu.be/abc'),
      (err) => {
        assert.ok(err instanceof Error);
        assert.ok(err.message.includes('422'));
        return true;
      }
    );
  });
});

// ---------------------------------------------------------------------------
// submitUrl — timeout
// ---------------------------------------------------------------------------
describe('createHttpClient — timeout', () => {
  let server;
  let client;

  before(async () => {
    server = await makeServer((_req, _res) => {
      // Never respond — simulate hang
    });
    client = createHttpClient(serverUrl(server), { timeoutMs: 50 });
  });

  after(async () => {
    await new Promise((r) => server.close(r));
  });

  test('throws on timeout', async () => {
    await assert.rejects(
      () => client.submitUrl('https://youtu.be/abc'),
      (err) => {
        assert.ok(err instanceof Error);
        // Client wraps AbortError as a timeout message, or the fetch itself throws AbortError
        const msg = err.message.toLowerCase();
        const isTimeout = msg.includes('timeout') || msg.includes('timed out') || msg.includes('abort') || err.name === 'AbortError';
        assert.ok(isTimeout, `Expected timeout/abort error, got: ${err.message}`);
        return true;
      }
    );
  });
});

// ---------------------------------------------------------------------------
// healthCheck
// ---------------------------------------------------------------------------
describe('createHttpClient — healthCheck', () => {
  test('returns true when /health returns 200', async () => {
    const server = await makeServer((_req, res) => {
      res.writeHead(200);
      res.end(JSON.stringify({ status: 'ok' }));
    });
    const client = createHttpClient(serverUrl(server));
    const healthy = await client.healthCheck();
    assert.equal(healthy, true);
    await new Promise((r) => server.close(r));
  });

  test('returns false when /health returns 500', async () => {
    const server = await makeServer((_req, res) => {
      res.writeHead(500);
      res.end('error');
    });
    const client = createHttpClient(serverUrl(server));
    const healthy = await client.healthCheck();
    assert.equal(healthy, false);
    await new Promise((r) => server.close(r));
  });

  test('returns false when server is unreachable', async () => {
    const client = createHttpClient('http://127.0.0.1:19999', { timeoutMs: 100 });
    const healthy = await client.healthCheck();
    assert.equal(healthy, false);
  });
});

// ---------------------------------------------------------------------------
// DEFAULT_TIMEOUT_MS export
// ---------------------------------------------------------------------------
test('DEFAULT_TIMEOUT_MS is a positive integer', () => {
  assert.equal(typeof DEFAULT_TIMEOUT_MS, 'number');
  assert.ok(DEFAULT_TIMEOUT_MS > 0);
});
