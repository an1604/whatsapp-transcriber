'use strict';

const { test, describe } = require('node:test');
const assert = require('node:assert/strict');

const { extractUrls } = require('../src/url_extractor');

// ---------------------------------------------------------------------------
// YouTube
// ---------------------------------------------------------------------------
describe('extractUrls — YouTube', () => {
  test('extracts standard watch URL', () => {
    const urls = extractUrls('Check this out: https://www.youtube.com/watch?v=dQw4w9WgXcQ');
    assert.equal(urls.length, 1);
    assert.equal(urls[0].platform, 'youtube');
    assert.ok(urls[0].url.includes('youtube.com'));
  });

  test('extracts short youtu.be URL', () => {
    const urls = extractUrls('Video: https://youtu.be/dQw4w9WgXcQ');
    assert.equal(urls.length, 1);
    assert.equal(urls[0].platform, 'youtube');
  });

  test('extracts YouTube Shorts URL', () => {
    const urls = extractUrls('https://www.youtube.com/shorts/abc123XYZ-_');
    assert.equal(urls.length, 1);
    assert.equal(urls[0].platform, 'youtube');
  });

  test('extracts multiple YouTube URLs from one message', () => {
    const urls = extractUrls(
      'First: https://youtu.be/aaa111 and second: https://www.youtube.com/watch?v=bbb222'
    );
    assert.equal(urls.length, 2);
    assert.ok(urls.every((u) => u.platform === 'youtube'));
  });

  test('deduplicates identical YouTube URLs', () => {
    const text = 'https://youtu.be/aaa111 and again: https://youtu.be/aaa111';
    const urls = extractUrls(text);
    assert.equal(urls.length, 1);
  });

  test('returns empty array when no YouTube URL present', () => {
    const urls = extractUrls('No video here, just text.');
    assert.equal(urls.length, 0);
  });
});

// ---------------------------------------------------------------------------
// Instagram
// ---------------------------------------------------------------------------
describe('extractUrls — Instagram', () => {
  test('extracts reel URL', () => {
    const urls = extractUrls('https://www.instagram.com/reel/ABC123xyz/');
    assert.equal(urls.length, 1);
    assert.equal(urls[0].platform, 'instagram');
  });

  test('extracts /p/ post URL', () => {
    const urls = extractUrls('https://www.instagram.com/p/ABC123/');
    assert.equal(urls.length, 1);
    assert.equal(urls[0].platform, 'instagram');
  });

  test('extracts IGTV URL', () => {
    const urls = extractUrls('Watch: https://www.instagram.com/tv/DEF456/');
    assert.equal(urls.length, 1);
    assert.equal(urls[0].platform, 'instagram');
  });
});

// ---------------------------------------------------------------------------
// TikTok
// ---------------------------------------------------------------------------
describe('extractUrls — TikTok', () => {
  test('extracts long-form TikTok URL', () => {
    const urls = extractUrls('https://www.tiktok.com/@user.name/video/1234567890123456789');
    assert.equal(urls.length, 1);
    assert.equal(urls[0].platform, 'tiktok');
  });

  test('extracts vm.tiktok.com short URL', () => {
    const urls = extractUrls('https://vm.tiktok.com/ZMxxxxxx/');
    assert.equal(urls.length, 1);
    assert.equal(urls[0].platform, 'tiktok');
  });
});

// ---------------------------------------------------------------------------
// Facebook
// ---------------------------------------------------------------------------
describe('extractUrls — Facebook', () => {
  test('extracts Facebook reel URL', () => {
    const urls = extractUrls('https://www.facebook.com/reel/1234567890');
    assert.equal(urls.length, 1);
    assert.equal(urls[0].platform, 'facebook');
  });

  test('extracts fb.watch short URL', () => {
    const urls = extractUrls('https://fb.watch/abc123XYZ/');
    assert.equal(urls.length, 1);
    assert.equal(urls[0].platform, 'facebook');
  });
});

// ---------------------------------------------------------------------------
// Mixed / edge cases
// ---------------------------------------------------------------------------
describe('extractUrls — mixed and edge cases', () => {
  test('extracts URLs from multiple platforms in one message', () => {
    const text = [
      'YouTube: https://youtu.be/yt123',
      'Instagram: https://www.instagram.com/reel/ig456/',
      'TikTok: https://vm.tiktok.com/tt789/',
    ].join(' ');
    const urls = extractUrls(text);
    assert.equal(urls.length, 3);
    const platforms = urls.map((u) => u.platform);
    assert.ok(platforms.includes('youtube'));
    assert.ok(platforms.includes('instagram'));
    assert.ok(platforms.includes('tiktok'));
  });

  test('returns empty array for empty string', () => {
    assert.deepEqual(extractUrls(''), []);
  });

  test('throws TypeError for non-string input', () => {
    assert.throws(() => extractUrls(null), TypeError);
    assert.throws(() => extractUrls(42), TypeError);
    assert.throws(() => extractUrls(undefined), TypeError);
  });

  test('ignores plain HTTP links that are not video platforms', () => {
    const urls = extractUrls('Visit https://example.com or https://google.com');
    assert.equal(urls.length, 0);
  });

  test('result objects have url and platform fields', () => {
    const urls = extractUrls('https://youtu.be/abc123');
    assert.ok('url' in urls[0]);
    assert.ok('platform' in urls[0]);
  });

  test('url field is a string', () => {
    const urls = extractUrls('https://youtu.be/abc123');
    assert.equal(typeof urls[0].url, 'string');
  });
});
