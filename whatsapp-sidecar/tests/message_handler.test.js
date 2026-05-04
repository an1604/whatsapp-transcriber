'use strict';

const { test, describe } = require('node:test');
const assert = require('node:assert/strict');

const { createMessageHandler } = require('../src/message_handler');
const { extractUrls } = require('../src/url_extractor');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeMockClient(overrides = {}) {
  const calls = [];
  return {
    submitUrl: async (url, sourceGroup) => {
      calls.push({ url, sourceGroup });
      if (overrides.rejectWith) throw overrides.rejectWith;
      return overrides.response ?? {
        video_id: 1,
        canonical_url: url,
        platform: 'youtube',
        is_duplicate: false,
      };
    },
    _calls: calls,
  };
}

function makeMsg(body, opts = {}) {
  return {
    body,
    isStatus: opts.isStatus ?? false,
    from: opts.from ?? 'group123@g.us',
    id: { remote: opts.remote ?? opts.from ?? 'group123@g.us' },
  };
}

// ---------------------------------------------------------------------------
// Basic forwarding
// ---------------------------------------------------------------------------
describe('createMessageHandler — basic forwarding', () => {
  test('forwards a YouTube URL to the pipeline', async () => {
    const client = makeMockClient();
    const handle = createMessageHandler(client, extractUrls, []);
    await handle(makeMsg('Watch: https://youtu.be/abc123'));
    assert.equal(client._calls.length, 1);
    assert.ok(client._calls[0].url.includes('youtu.be'));
  });

  test('forwards multiple URLs from one message', async () => {
    const client = makeMockClient();
    const handle = createMessageHandler(client, extractUrls, []);
    await handle(makeMsg(
      'https://youtu.be/aaa111 and https://youtu.be/bbb222'
    ));
    assert.equal(client._calls.length, 2);
  });

  test('passes source_group from message', async () => {
    const client = makeMockClient();
    const handle = createMessageHandler(client, extractUrls, []);
    await handle(makeMsg('https://youtu.be/abc', { remote: 'mygroup@g.us' }));
    assert.equal(client._calls[0].sourceGroup, 'mygroup@g.us');
  });

  test('does nothing for message with no video URLs', async () => {
    const client = makeMockClient();
    const handle = createMessageHandler(client, extractUrls, []);
    await handle(makeMsg('Hello, how are you?'));
    assert.equal(client._calls.length, 0);
  });

  test('does nothing for empty message body', async () => {
    const client = makeMockClient();
    const handle = createMessageHandler(client, extractUrls, []);
    await handle(makeMsg(''));
    assert.equal(client._calls.length, 0);
  });

  test('skips status messages', async () => {
    const client = makeMockClient();
    const handle = createMessageHandler(client, extractUrls, []);
    await handle(makeMsg('https://youtu.be/abc', { isStatus: true }));
    assert.equal(client._calls.length, 0);
  });
});

// ---------------------------------------------------------------------------
// Community ID filtering
// ---------------------------------------------------------------------------
describe('createMessageHandler — community filtering', () => {
  test('allows message from matching community ID', async () => {
    const client = makeMockClient();
    const handle = createMessageHandler(client, extractUrls, ['allowed@g.us']);
    await handle(makeMsg('https://youtu.be/abc', { remote: 'allowed@g.us' }));
    assert.equal(client._calls.length, 1);
  });

  test('blocks message from non-matching community', async () => {
    const client = makeMockClient();
    const handle = createMessageHandler(client, extractUrls, ['allowed@g.us']);
    await handle(makeMsg('https://youtu.be/abc', { remote: 'other@g.us' }));
    assert.equal(client._calls.length, 0);
  });

  test('allows all communities when communityIds is empty', async () => {
    const client = makeMockClient();
    const handle = createMessageHandler(client, extractUrls, []);
    await handle(makeMsg('https://youtu.be/abc', { remote: 'anyone@g.us' }));
    assert.equal(client._calls.length, 1);
  });

  test('blocks when communityIds list has no matching entry', async () => {
    const client = makeMockClient();
    const handle = createMessageHandler(client, extractUrls, ['a@g.us', 'b@g.us']);
    await handle(makeMsg('https://youtu.be/abc', { remote: 'c@g.us' }));
    assert.equal(client._calls.length, 0);
  });
});

// ---------------------------------------------------------------------------
// Error handling — pipeline client failures should not crash
// ---------------------------------------------------------------------------
describe('createMessageHandler — error resilience', () => {
  test('does not throw when pipeline client rejects', async () => {
    const client = makeMockClient({ rejectWith: new Error('connection refused') });
    const handle = createMessageHandler(client, extractUrls, []);
    await assert.doesNotReject(() =>
      handle(makeMsg('https://youtu.be/abc'))
    );
  });

  test('continues processing remaining URLs after one failure', async () => {
    let callCount = 0;
    const client = {
      submitUrl: async () => {
        callCount++;
        if (callCount === 1) throw new Error('first fails');
        return { video_id: callCount, canonical_url: 'x', platform: 'youtube', is_duplicate: false };
      },
      _calls: [],
    };
    const handle = createMessageHandler(client, extractUrls, []);
    await handle(makeMsg(
      'https://youtu.be/first111 and https://youtu.be/second222'
    ));
    assert.equal(callCount, 2, 'Should attempt both URLs');
  });

  test('handles duplicate response gracefully', async () => {
    const client = makeMockClient({
      response: { video_id: 5, canonical_url: 'x', platform: 'youtube', is_duplicate: true },
    });
    const handle = createMessageHandler(client, extractUrls, []);
    await assert.doesNotReject(() =>
      handle(makeMsg('https://youtu.be/abc'))
    );
  });
});

// ---------------------------------------------------------------------------
// Custom extractUrls
// ---------------------------------------------------------------------------
describe('createMessageHandler — custom extractUrls', () => {
  test('uses provided extractUrls function', async () => {
    const client = makeMockClient();
    const customExtract = () => [{ url: 'https://custom.com/video/1', platform: 'youtube' }];
    const handle = createMessageHandler(client, customExtract, []);
    await handle(makeMsg('anything'));
    assert.equal(client._calls.length, 1);
    assert.equal(client._calls[0].url, 'https://custom.com/video/1');
  });

  test('handles extractUrls returning empty array', async () => {
    const client = makeMockClient();
    const handle = createMessageHandler(client, () => [], []);
    await handle(makeMsg('https://youtu.be/abc'));
    assert.equal(client._calls.length, 0);
  });
});
