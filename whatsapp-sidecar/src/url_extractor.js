'use strict';

// Patterns for supported video platforms
const PLATFORM_PATTERNS = [
  // YouTube
  { platform: 'youtube', re: /https?:\/\/(?:www\.)?(?:youtube\.com\/(?:watch\?v=|shorts\/|embed\/)|youtu\.be\/)[\w-]+/g },
  // Instagram
  { platform: 'instagram', re: /https?:\/\/(?:www\.)?instagram\.com\/(?:reel|p|tv|reels)\/[\w-]+\/?/g },
  // TikTok
  { platform: 'tiktok', re: /https?:\/\/(?:www\.|vm\.|vt\.)?tiktok\.com\/(?:@[\w.]+\/video\/\d+|[\w]+)\/?/g },
  // Facebook - full URLs with recognized path prefixes
  { platform: 'facebook', re: /https?:\/\/(?:www\.|m\.)?facebook\.com\/(?:reel\/|watch\/\?v=|[\w.]+\/videos\/|video\.php\?v=|share\/r\/|share\/v\/)[\w?=&.-]*/g },
  // Facebook - fb.watch short URLs (any path segment)
  { platform: 'facebook', re: /https?:\/\/fb\.watch\/[\w-]+\/?/g },
];

/**
 * Extract all supported video URLs from a text string.
 * Returns an array of { url, platform } objects (deduplicated by URL).
 *
 * @param {string} text
 * @returns {{ url: string, platform: string }[]}
 */
function extractUrls(text) {
  if (typeof text !== 'string') {
    throw new TypeError(`extractUrls: expected string, got ${typeof text}`);
  }

  const seen = new Set();
  const results = [];

  for (const { platform, re } of PLATFORM_PATTERNS) {
    // Reset lastIndex for global regexes
    re.lastIndex = 0;
    let match;
    while ((match = re.exec(text)) !== null) {
      const url = match[0];
      if (!seen.has(url)) {
        seen.add(url);
        results.push({ url, platform });
      }
    }
  }

  return results;
}

module.exports = { extractUrls, PLATFORM_PATTERNS };
